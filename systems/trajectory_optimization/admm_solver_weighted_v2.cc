
#include "drake/systems/trajectory_optimization/admm_solver_weighted_v2.h"

typedef Array<bool,Dynamic,1> ArrayXb;

using namespace std::chrono;
using namespace std::placeholders;

namespace drake {
    namespace systems {
        namespace trajectory_optimization {
            namespace admm_solver {
                
                AdmmSolverWeightedV2::AdmmSolverWeightedV2(systems::System<double>* par_system, Eigen::VectorXd par_x0, Eigen::VectorXd par_xf, double par_T, int par_N, int par_max_iter) : AdmmSolverBase(par_system, par_x0, par_xf, par_T, par_N, par_max_iter) {
                }
                
                AdmmSolverWeightedV2::AdmmSolverWeightedV2(systems::System<double>* par_system) : AdmmSolverBase(par_system) {}
                
                
                /* ---------------------------------------------- SOLVE METHOD ---------------------------------------------- */
                
                std::string AdmmSolverWeightedV2::solve(Eigen::Ref<Eigen::VectorXd> y) {
                    
                    // define clean output format for vectors and matrices
                    IOFormat CleanFmt(6, 0, ", ", "\n");
                    
                    // open file for writing output
                    ofstream output_stream;
                    output_stream.open(output_file);
                    if (!output_stream.is_open()) {
                        std::cerr << "Problem opening output file.";
                        return 0;
                    }
                    
                    // open file for writing trajectories
                    ofstream traj_stream;
                    traj_stream.open(traj_file);
                    if (!traj_stream.is_open()) {
                        std::cerr << "Problem opening output file.";
                        return 0;
                    }
                    
                    // allocate array for x, y, lambda
                    Eigen::VectorXd x(y);
                    Eigen::VectorXd y_prev(y);
                    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(N * (num_states + num_inputs));
                    
                    // initialize solution vectors
                    solution_x = Eigen::MatrixXd::Zero(num_states, N);
                    solution_u = Eigen::MatrixXd::Zero(num_inputs, N);
                    
                    // allocate arrays for f's, A's, and B's
                    Eigen::VectorXd* f = new Eigen::VectorXd[N];
                    Eigen::MatrixXd* A = new Eigen::MatrixXd[N];
                    Eigen::MatrixXd* B = new Eigen::MatrixXd[N];
                    
                    for (int i = 0; i < N; i++) {
                        A[i] = Eigen::MatrixXd::Zero(num_states, num_states);
                        B[i] = Eigen::MatrixXd::Zero(num_states, num_inputs);
                        f[i] = Eigen::VectorXd::Zero(num_states);
                    }
                    
                    // allocate array for M
                    std::vector<Triplet<double> > tripletsM;
                    tripletsM.reserve(2 * (N+1) * num_states * num_states + 2 * (N-1) * num_states * num_inputs);
                    Eigen::SparseMatrix<double> M((N+1) * num_states, N * (num_states + num_inputs));
                    
                    // allocate array for G
                    std::vector<Triplet<double> > tripletsG;
                    Eigen::SparseMatrix<double> G(N * num_constraints, N * (num_states + num_inputs));
                    
                    // allocate array for c
                    Eigen::VectorXd c = Eigen::VectorXd::Zero((N+1) * num_states);
                    
                    // allocate array for g and h
                    Eigen::VectorXd g = Eigen::VectorXd::Zero(num_constraints * N);
                    Eigen::VectorXd h = Eigen::VectorXd::Zero(num_constraints * N);
                    Eigen::VectorXd g_unweighted = Eigen::VectorXd::Zero(num_constraints * N);
                    
                    // vector of weights
                    Eigen::VectorXd weights = Eigen::VectorXd::Ones(num_constraints * N);
                    
                    // allocate memory for midpoint values
                    Eigen::VectorXd mid_state(num_states);
                    Eigen::VectorXd mid_input(num_inputs);
                    Eigen::VectorXd temp;
                    Eigen::VectorXd feasibilityVector = Eigen::VectorXd::Zero((N+1) * num_states);
                    //Eigen::VectorXd constraintVector = Eigen::VectorXd::Zero((N+1) * num_states);
                    Eigen::VectorXd constraintVector = Eigen::VectorXd::Zero(N * num_constraints);
                    double objective;
                    double oldObjective;
                    double feasibilityNorm;
                    double oldFeasibilityNorm;
                    double constraintNorm;
                    double full_objective;
                    bool not_feasible = true;
                    bool not_converged = true;
                    
                    // store all costs in the cost matrix R and the cost vector b
                    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(N * (num_states + num_inputs), N * (num_states + num_inputs));
                    Eigen::VectorXd b = Eigen::VectorXd::Zero(N * (num_states + num_inputs));
                    
                    // R will be a diagonal block matrix of Q's and R's
                    for (int i = 0; i < N; i++) {
                        R.block(i * num_states, i * num_states, num_states, num_states) = costQ;
                        R.block(N * num_states + i * num_inputs, N * num_states + i * num_inputs, num_inputs, num_inputs) = costR;
                        
                        b.segment(i * num_states, num_states) = costq;
                        b.segment(N * num_states + i * num_inputs, num_inputs) = costr;
                    }
                    
                    // the cost on the final state is the running cost (Q) added to the final cost (Qf)
                    R.block((N-1) * num_states, (N-1) * num_states, num_states, num_states) = R.block((N-1) * num_states, (N-1) * num_states, num_states, num_states) + costQf;
                    b.segment((N-1) * num_states, num_states) = b.segment((N-1) * num_states, num_states) + costqf;
                    
                    // initialize timers
                    std::chrono::system_clock::time_point admm_update1_time_start;
                    std::chrono::system_clock::time_point admm_update1_time_end;
                    std::chrono::system_clock::time_point admm_update2_time_start;
                    std::chrono::system_clock::time_point admm_update2_time_end;
                    std::chrono::system_clock::time_point admm_dynamics_time_start;
                    std::chrono::system_clock::time_point admm_dynamics_time_end;
                    double admm_update1_timer = 0.0;
                    double admm_update2_timer = 0.0;
                    double admm_dynamics_timer = 0.0;
                    
                    // outer ADMM iterations
                    Eigen::VectorXd point_g = Eigen::VectorXd::Zero(num_constraints);
                    Eigen::MatrixXd point_dg_x = Eigen::MatrixXd::Zero(num_constraints, num_states);
                    Eigen::MatrixXd point_dg_u = Eigen::MatrixXd::Zero(num_constraints, num_inputs);
                    
                    // set rho's to initial values
                    double rho1 = initial_rho1;
                    double rho2 = initial_rho2;
                    double rho3 = initial_rho3;
                    
                    // write a header
                    output_stream << "\n---------------------------------\n";
                    output_stream << "STARTING SOLVE:\n";
                    output_stream << "---------------------------------\n";
                    output_stream << "num states (in ADMM): " << num_states << "\n";
                    output_stream << "num inputs (in ADMM): " << num_inputs << "\n";
                    output_stream << "num constraints per point = " << num_constraints << "\n";
                    output_stream << "constraints list length = " << single_constraints_list.size() << "\n\n";
                    
                    // iterations counter
                    int i = 0;
                    
                    while (i < max_iter && (not_feasible || not_converged)) {
                        
                        // write y to file
                        if (DEBUG) {
                            traj_stream << y.transpose() << "\n";
                        }
                        
                        // -- for each point, calculate and fill in linearized dynamics within the M matrix and c vector -- //
                        
                        for (int ii = 0; ii < N-1; ii++) {
                            
                            // find midpoint state
                            for (int iii=0; iii < num_states; iii++) {
                                mid_state[iii] = 0.5 * (getStateFromY(y, ii, iii) + getStateFromY(y, ii+1, iii));
                            }
                            
                            // find midpoint input
                            for (int iii=0; iii < num_inputs; iii++) {
                                mid_input[iii] = 0.5 * (getInputFromY(y, ii, iii) + getInputFromY(y, ii+1, iii));
                            }
                            
                            // set x and u
                            auto autodiff_args = math::initializeAutoDiffTuple(mid_state, mid_input);
                            
                            autodiff_context->get_mutable_continuous_state_vector().SetFromVector(std::get<0>(autodiff_args));
                            autodiff_context->FixInputPort(0, std::get<1>(autodiff_args));
                            
                            // calculate f and df from dynamics function pointer
                            admm_dynamics_time_start = std::chrono::system_clock::now();
                        
                            // calculate first- and second- order derivatives
                            std::unique_ptr<ContinuousState<AutoDiffXd>> autodiff_xdot = autodiff_system->AllocateTimeDerivatives();
                            autodiff_system->CalcTimeDerivatives(*autodiff_context, autodiff_xdot.get());
                            auto autodiff_xdot_vec = autodiff_xdot->CopyToVector();
                            
                            // get first- and second- order derivatives
                            f[ii] = math::autoDiffToValueMatrix(autodiff_xdot_vec);
                            const Eigen::MatrixXd AB = math::autoDiffToGradientMatrix(autodiff_xdot_vec);
                            A[ii] = AB.leftCols(num_states);
                            B[ii] = AB.rightCols(num_inputs);
                            
                            // end timer
                            admm_dynamics_time_end = std::chrono::system_clock::now();
                            admm_dynamics_timer = admm_dynamics_timer + (duration_cast<duration<double>>(admm_dynamics_time_end - admm_dynamics_time_start)).count();
                        
                            // put in correct place in large M matrix
                            placeAinM(&tripletsM, A[ii], ii);
                            placeBinM(&tripletsM, B[ii], ii);
                            
                            // make right-hand-side vector for this time point
                            makeCVector(c, ii, mid_state, mid_input, f[ii], A[ii], B[ii], y);
                        }
                        
                        // fill in identity in M and x0, xf in c
                        for (int ii = 0; ii < num_states; ii++) {
                            c[ii] = x0[ii];
                            c[N * num_states + ii] = xf[ii];
                            
                            tripletsM.push_back(Triplet<double>(ii, ii, 1));
                            tripletsM.push_back(Triplet<double>(N * num_states + ii, (N-1) * num_states + ii, 1));
                        }
                        
                        M.setFromTriplets(tripletsM.begin(), tripletsM.end());
                        
                        
                        // -- iterate, calculate, and fill in all constraints that are functions of single points -- //
                        
                        int running_constraint_counter = 0;
                        for (int iii = 0; iii < int(single_constraints_list.size()); iii++) {
                            // get constraint function
                            single_constraint_struct cf = single_constraints_list[iii];
                            
                            // iterate and apply constraint to all points
                            for (int ii = 0; ii < N; ii++) {
                                // get current state/input
                                Eigen::VectorXd state = y.segment(ii * num_states, num_states);
                                Eigen::VectorXd input = y.segment(N * num_states + ii * num_inputs, num_inputs);
                                
                                // prepare matrices (OPTIMIZE LATER)
                                Eigen::VectorXd single_g = Eigen::VectorXd::Zero(cf.length);
                                Eigen::MatrixXd single_dg_x = Eigen::MatrixXd::Zero(cf.length, num_states);
                                Eigen::MatrixXd single_dg_u = Eigen::MatrixXd::Zero(cf.length, num_inputs);
                                
                                // evaluate constraints
                                cf.function(ii, state, input, single_g, single_dg_x, single_dg_u);
                                
                                // zero out all unviolated inequality constraints
                                if (cf.flag == INEQUALITY) {
                                    for (int iiii = 0; iiii < cf.length; iiii++) {
                                        if (single_g[iiii] <= 0) {
                                            single_g[iiii] = 0;
                                            single_dg_x.block(iiii, 0, 1, num_states) = 0 * single_dg_x.block(iiii, 0, 1, num_states);
                                            single_dg_u.block(iiii, 0, 1, num_inputs) = 0 * single_dg_u.block(iiii, 0, 1, num_inputs);
                                        }
                                    }
                                }
                                
                                Eigen::VectorXd single_g_unweighted(single_g);
                                
                                // get weights vector for these constraints
                                Eigen::VectorXd current_weights = weights.segment(ii * num_constraints + running_constraint_counter, cf.length);
                                
                                // multiply constraint values by weights
                                single_g = single_g.cwiseProduct(current_weights);
                                single_dg_x = current_weights.asDiagonal() * single_dg_x;
                                single_dg_u = current_weights.asDiagonal() * single_dg_u;
                                
                                // for each constraint, check if constraint is active and update weights in that case
                                for (int iv = 0; iv < cf.length; iv++) {
                                    // index of constraint
                                    int index = ii * num_constraints + running_constraint_counter + iv;
                                    
                                    // if constraint is violated, increase weight
                                    if (((cf.flag == INEQUALITY) & (single_g[iv] > 0)) || ((cf.flag == EQUALITY) & (single_g[iv] != 0))) {
                                        weights[index] = min(weights[index] * rho3_increase_rate, 1e9/rho3);
                                    }
                                }
                                
                                // place constraints in the G matrix and g vector
                                g.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g;
                                placeinG(&tripletsG, single_dg_x, single_dg_u, ii, running_constraint_counter, cf.length);
                                g_unweighted.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g_unweighted;
                            }
                            
                            running_constraint_counter = running_constraint_counter + cf.length;
                        }
                        
                        
                        // -- iterate, calculate, and fill in all constraints that are functions of two consecutive points -- //
                        
                        output_stream << "\n";
                        
                        for (int iii = 0; iii < int(double_constraints_list.size()); iii++) {
                            // get constraint function
                            double_constraint_struct cf = double_constraints_list[iii];
                            
                            for (int ii = 0; ii < N-1; ii++) {
                                // get current state/input
                                Eigen::VectorXd state1 = y.segment(ii * num_states, num_states);
                                Eigen::VectorXd input1 = y.segment(N * num_states + ii * num_inputs, num_inputs);
                                
                                // get next state/input
                                Eigen::VectorXd state2 = y.segment((ii+1) * num_states, num_states);
                                Eigen::VectorXd input2 = y.segment(N * num_states + (ii+1) * num_inputs, num_inputs);
                                
                                // prepare matrices (OPTIMIZE LATER)
                                Eigen::VectorXd single_g = Eigen::VectorXd::Zero(cf.length);
                                Eigen::MatrixXd single_dg_x1 = Eigen::MatrixXd::Zero(cf.length, num_states);
                                Eigen::MatrixXd single_dg_u1 = Eigen::MatrixXd::Zero(cf.length, num_inputs);
                                Eigen::MatrixXd single_dg_x2 = Eigen::MatrixXd::Zero(cf.length, num_states);
                                Eigen::MatrixXd single_dg_u2 = Eigen::MatrixXd::Zero(cf.length, num_inputs);
                                
                                // evaluate constraints
                                cf.function(ii, state1, input1, state2, input2, single_g, single_dg_x1, single_dg_u1, single_dg_x2, single_dg_u2);
                                
                                if (cf.flag == INEQUALITY) {
                                    for (int iiii = 0; iiii < cf.length; iiii++) {
                                        if (single_g[iiii] <= 0) {
                                            single_g[iiii] = 0;
                                            single_dg_x1.block(iiii, 0, 1, num_states) = 0 * single_dg_x1.block(iiii, 0, 1, num_states);
                                            single_dg_u1.block(iiii, 0, 1, num_inputs) = 0 * single_dg_u1.block(iiii, 0, 1, num_inputs);
                                            single_dg_x2.block(iiii, 0, 1, num_states) = 0 * single_dg_x2.block(iiii, 0, 1, num_states);
                                            single_dg_u2.block(iiii, 0, 1, num_inputs) = 0 * single_dg_u2.block(iiii, 0, 1, num_inputs);
                                        }
                                    }
                                }
                                
                                // make an unweighted copy of the constraints
                                Eigen::VectorXd single_g_unweighted(single_g);
                                
                                // select current weights
                                Eigen::VectorXd current_weights = weights.segment(ii * num_constraints + running_constraint_counter, cf.length);
                                
                                // make element-wise multiply of array values by weights
                                single_g = single_g.cwiseProduct(current_weights);
                                single_dg_x1 = current_weights.asDiagonal() * single_dg_x1;
                                single_dg_u1 = current_weights.asDiagonal() * single_dg_u1;
                                single_dg_x2 = current_weights.asDiagonal() * single_dg_x2;
                                single_dg_u2 = current_weights.asDiagonal() * single_dg_u2;
                                
                                // for each constraint, check if constraint is active and update weights in that case
                                for (int iv = 0; iv < cf.length; iv++) {
                                    // index of constraint
                                    int index = ii * num_constraints + running_constraint_counter + iv;
                                    // if constraint is violated, increase weight
                                    if (((cf.flag == INEQUALITY) & (single_g[iv] > 0)) || ((cf.flag == EQUALITY) & (single_g[iv] != 0))) {
                                        weights[index] = min(weights[index] * rho3_increase_rate, 1e9/rho3);
                                    }
                                }
                                
                                // place constraints in the G matrix and g vector
                                g.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g;
                                placeinG(&tripletsG, single_dg_x1, single_dg_u1, ii, running_constraint_counter, cf.length);
                                placeinG(&tripletsG, single_dg_x2, single_dg_u2, ii+1, running_constraint_counter, cf.length);
                                
                                g_unweighted.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g_unweighted;
                                
                                output_stream << single_g.lpNorm<Eigen::Infinity>() << "  ";
                            }
                            
                            running_constraint_counter = running_constraint_counter + cf.length;
                        }
                        
                        output_stream << "\n\n";
                        
                        // make sparse G matrix, holding all linearized constraints
                        G.setFromTriplets(tripletsG.begin(), tripletsG.end());
                        
                        h = G * y - g;
                        
                        // calculate values for "this" iteration (before the update)
                        objective = y.transpose() * R * y;
                        feasibilityVector = M * y - c;
                        feasibilityNorm = feasibilityVector.lpNorm<Eigen::Infinity>();
                        constraintVector = g_unweighted; //previous: constraintVector = G * y - h;
                        constraintNorm = constraintVector.lpNorm<Eigen::Infinity>();
                        full_objective = objective + rho2 * pow(feasibilityVector.lpNorm<2>(), 2) + rho3 * pow(constraintVector.lpNorm<2>(), 2);
                        
                        not_feasible = (feasibilityVector.lpNorm<Eigen::Infinity>() > tol_feasibility || constraintVector.lpNorm<Eigen::Infinity>() > tol_constraints);
                        not_converged = (abs(oldObjective - objective)/objective > tol_objective);
                        //not_converged = false;
                        
                        if (!not_feasible & !not_converged) {
                            break;
                        }
                        
                        // first ADMM update
                        admm_update1_time_start = std::chrono::system_clock::now(); // start timer
                        
                        x = proximalUpdateObjective(y, lambda, R, b, rho1);
                        
                        admm_update1_time_end = std::chrono::system_clock::now(); // end timer
                        admm_update1_timer = admm_update1_timer + (duration_cast<duration<double>>(admm_update1_time_end - admm_update1_time_start)).count();
                        
                        // second ADMM update
                        admm_update2_time_start = std::chrono::system_clock::now(); // start timer
                        
                        y = proximalUpdateConstraints(x, lambda, M, c, G, h, rho1, rho2, rho3);
                        
                        admm_update2_time_end = std::chrono::system_clock::now(); // end timer
                        admm_update2_timer = admm_update2_timer + (duration_cast<duration<double>>(admm_update2_time_end - admm_update2_time_start)).count();
                        
                        // dual update
                        lambda = lambda + rho1 * (x - y);
                        
                        // increase rho2 and rho3
                        /*
                        objective = y.transpose() * R * y;
                        feasibilityVector = M * y - c;
                        feasibilityNorm = feasibilityVector.lpNorm<Eigen::Infinity>();
                        constraintVector = G * y - h; //g; //previous: constraintVector = G * y - h;
                        constraintNorm = constraintVector.lpNorm<Eigen::Infinity>();
                        full_objective = objective + rho2 * pow(feasibilityVector.lpNorm<2>(), 2) + rho3 * pow(constraintVector.lpNorm<2>(), 2);
                        
                        not_feasible = (feasibilityVector.lpNorm<Eigen::Infinity>() > tol_feasibility || constraintVector.lpNorm<Eigen::Infinity>() > tol_constraints);
                        not_converged = (abs(oldObjective - objective)/objective > tol_objective);
                         */
                        
                        // increase rho2
                        rho2 = min(rho2 * rho2_increase_rate, rho_max);
                        rho3 = min(rho3 * rho3_increase_rate, rho_max);
                        
                        // decrease rho1 if constraints are mostly satisfied
                        if (i > 100) {
                            rho1 = max(rho1_min, rho1 / rho1_decrease_rate);
                        }
                        
                        // print / compute info
                        if (i % 1 == 0) {
                            output_stream << "Iteration " << i << " -- objective cost: " << objective << " -- feasibility (inf-norm): " << feasibilityNorm << " -- constraint satisfaction (inf-norm): " << constraintNorm << " -- full objective: " << full_objective << " -- G norm: " << G.norm() << " -- rho0: " << rho1 << " -- rho1: " << rho2 << " -- rho2: " << rho3 << "\n";
                        }
                        
                        tripletsM.clear();
                        tripletsG.clear();
                        oldFeasibilityNorm = feasibilityNorm;
                        oldObjective = objective;
                        
                        i++;
                    }
                    
                    // problem:
                    // constraint calculations apply to previous point, and there is oscillation between two trajectories
                    // write visualization of trajectory (as a video)
                    //
                    
                    // print final convergence output
                    output_stream << "Finished at iteration " << i << " -- objective cost: " << objective << " -- feasibility (inf-norm): " << feasibilityNorm << " -- constraint satisfaction (inf-norm): " << constraintNorm << "\n";
                    
                    // fill in solution
                    for (int ii = 0; ii < N; ii++) {
                        solution_x.block(0, ii, num_states, 1) = y.segment(ii * num_states, num_states);
                        solution_u.block(0, ii, num_inputs, 1) = y.segment(N * num_states + ii * num_inputs, num_inputs);
                    }
                    
                    // print timing information
                    output_stream << "\n---------------------------------\n";
                    output_stream << "TIMING:\n";
                    output_stream << "---------------------------------\n";
                    output_stream << "Total time spent in dynamics: " << admm_dynamics_timer << " (sec)\n";
                    output_stream << "Total time spent in first ADMM update: " << admm_update1_timer << " (sec)\n";
                    output_stream << "Total time spent in second ADMM update: " << admm_update2_timer << " (sec)\n";
                    output_stream << "Total update time: " << admm_update1_timer + admm_update2_timer << " (sec)\n";
                    solve_flag = true;
                    
                    // close streams
                    output_stream.close();
                    traj_stream.close();
                    
                    // return
                    num_latest_iterations = i;
                    if (i == max_iter) {
                        return "IterationLimit";
                    } else if (isnan(objective)) {
                        return "DivergenceToNaN";
                    } else {
                        return "SolutionFound";
                    }
                }
                
                // SECOND PROXIMAL UPDATE
                Eigen::VectorXd AdmmSolverWeightedV2::proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> lambda, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3) {
                    Eigen::MatrixXd Mt = M.transpose();
                    Eigen::MatrixXd Gt = G.transpose();
                    
                    Eigen::MatrixXd temp = rho1 * Eigen::MatrixXd::Identity(M.cols(), M.cols()) + rho2 * Mt * M + rho3 * Gt * G;
                    return temp.llt().solve(lambda + rho1 * x + rho2 * Mt * c + rho3 * Gt * h);
                }
            } // namespace admm_solver
        } // namespace trajectory_optimization
    } // namespace systems
} // namespace drake




/*
 Eigen::VectorXd AdmmSolverWeightedV2::proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> y, Eigen::Ref<Eigen::VectorXd> lambda, Eigen::Ref<Eigen::MatrixXd> R, Eigen::Ref<Eigen::VectorXd> b, double rho1) {
 Eigen::MatrixXd temp = 2 * R + rho1 * Eigen::MatrixXd::Identity(N * (num_states + num_inputs), N * (num_states + num_inputs));
 return temp.llt().solve(rho1 * y - lambda - b);
 } */

/*
 Eigen::VectorXd AdmmSolverWeightedV2::proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> y, Eigen::Ref<Eigen::VectorXd> lambda, Eigen::Ref<Eigen::MatrixXd> R, double rho1) {
 Eigen::MatrixXd temp = 2 * R + rho1 * Eigen::MatrixXd::Identity(N * (num_states + num_inputs), N * (num_states + num_inputs));
 return temp.llt().solve(rho1 * (y - lambda) - b);
 }
 */


/*
 Eigen::VectorXd AdmmSolverWeightedV2::proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> lambda, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3) {
 Eigen::MatrixXd Mt = M.transpose();
 Eigen::MatrixXd Gt = G.transpose();
 
 Eigen::MatrixXd temp = rho1 * Eigen::MatrixXd::Identity(M.cols(), M.cols()) + 2 * rho2 * Mt * M + 2 * rho3 * Gt * G;
 return temp.llt().solve(rho1 * (lambda + x) + 2 * rho2 * Mt * c + 2 * rho3 * Gt * h);
 }
 */
