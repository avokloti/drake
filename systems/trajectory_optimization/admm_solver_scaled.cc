// Version with obstacle constraints(and sparse M and G)!

// run as: g++ -I /usr/local/include/eigen3/ -O2 solveADMMconstraints.cpp

#include "drake/systems/trajectory_optimization/admm_solver_scaled.h"

typedef Array<bool,Dynamic,1> ArrayXb;

using namespace std::chrono;
using namespace std::placeholders;

#define SCALING 1

namespace drake {
    namespace systems {
        namespace trajectory_optimization {
            namespace admm_solver {
                
                AdmmSolverScaled::AdmmSolverScaled(systems::System<double>* par_system, Eigen::VectorXd par_x0, Eigen::VectorXd par_xf, double par_T, int par_N, int par_max_iter) : AdmmSolverBase(par_system, par_x0, par_xf, par_T, par_N, par_max_iter) {
                }
                
                AdmmSolverScaled::AdmmSolverScaled(systems::System<double>* par_system) : AdmmSolverBase(par_system) {}
                
                
                /* ---------------------------------------------- SOLVE METHOD ---------------------------------------------- */
                
                std::string AdmmSolverScaled::solve(Eigen::Ref<Eigen::VectorXd> y) {
                    
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
                    
                    // vector of weights
                    Eigen::VectorXd weights = Eigen::VectorXd::Ones(num_constraints * N);
                    Eigen::VectorXd constraint_scale_factor = Eigen::VectorXd::Ones(num_constraints * N);
                    double M_scale_factor;
                    double G_scale_factor;
                    
                    // allocate memory for midpoint values
                    Eigen::VectorXd mid_state(num_states);
                    Eigen::VectorXd mid_input(num_inputs);
                    Eigen::VectorXd temp;
                    Eigen::VectorXd scaledFeasibilityVector = Eigen::VectorXd::Zero((N+1) * num_states);
                    Eigen::VectorXd unscaledConstraintVector = Eigen::VectorXd::Zero((N+1) * num_states);
                    double scaledObjective;
                    double unscaledObjective;
                    double oldObjective;
                    double scaledFeasibilityNorm;
                    double oldFeasibilityNorm;
                    double unscaledConstraintNorm;
                    double fullObjective;
                    
                    bool not_feasible = true;
                    bool not_converged = true;
                    
                    // cost matrix
                    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(N * (num_states + num_inputs), N * (num_states + num_inputs));
                    Eigen::VectorXd b = Eigen::VectorXd::Zero(N * (num_states + num_inputs));
                    for (int i = 0; i < N; i++) {
                        R.block(i * num_states, i * num_states, num_states, num_states) = costQ;
                        R.block(N * num_states + i * num_inputs, N * num_states + i * num_inputs, num_inputs, num_inputs) = costR;
                        
                        b.segment(i * num_states, num_states) = costq;
                        b.segment(N * num_states + i * num_inputs, num_inputs) = costr;
                    }
                    //R.block((N-1) * num_states, (N-1) * num_states, num_states, num_states) = R.block((N-1) * num_states, (N-1) * num_states, num_states, num_states) + costQf;
                    //b.segment((N-1) * num_states, num_states) = b.segment((N-1) * num_states, num_states) + costqf;
                    R.block((N-1) * num_states, (N-1) * num_states, num_states, num_states) = costQf;
                    b.segment((N-1) * num_states, num_states) = costqf;
                    
                    // scale R?
                    std::cout << "R scale factor: " << R.norm() << std::endl;
                    
                    // scale R and b (once) and save scale factor
                    double R_scale_factor;
                    if (SCALING) {
                        R_scale_factor = R.norm();
                    } else {
                        R_scale_factor = 1;
                    }
                    R = R / R_scale_factor;
                    b = b / R_scale_factor;
                    
                    IOFormat CleanFmt(6, 0, ", ", "\n");
                    
                    // initialize times
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
                    
                    // update rho bounds
                    rho1_min = rho1 * 0.01;
                    rho_max = std::max(rho2, rho3) * 1e4;
                    
                    // open output file for writing y
                    ofstream output_stream;
                    output_stream.open(output_file);
                    if (!output_stream.is_open()) {
                        std::cerr << "Problem opening output file.";
                        return 0;
                    }
                    
                    ofstream traj_stream;
                    traj_stream.open(traj_file);
                    if (!traj_stream.is_open()) {
                        std::cerr << "Problem opening output file.";
                        return 0;
                    }
                    
                    
                    output_stream << "\n---------------------------------\n";
                    output_stream << "STARTING SOLVE:\n";
                    output_stream << "---------------------------------\n";
                    output_stream << "num states (in ADMM): " << num_states << "\n";
                    output_stream << "num inputs (in ADMM): " << num_inputs << "\n";
                    output_stream << "num constraints per point = " << num_constraints << "\n";
                    output_stream << "constraints list length = " << single_constraints_list.size() << "\n\n";
                    
                    int i = 0;
                    while (i < max_iter && (not_feasible || not_converged)) {
                    //while (i < 250) {
                        
                        // write y to file
                        if (DEBUG) {
                            traj_stream << y.transpose() << "\n";
                        }
                        
                        // for each time point
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
                            
                            // make RHS vector for this time point
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
                        
                        // scale M and c (once) and save scale factor
                        if (i == 0) {
                            if (SCALING) {
                                M_scale_factor = M.norm();
                            } else {
                                M_scale_factor = 1;
                            }
                            std::cout << "M scale factor: " << M.norm() << std::endl;
                        }
                        M = M / M_scale_factor;
                        c = c / M_scale_factor;
                        
                        //std::cout << "M norm: " << M.norm() << std::endl;
                        /*
                        if (M.norm() > 1) {
                            M = M / M.norm();
                            c = c / M.norm();
                        } */
                        
                        int running_constraint_counter = 0;
                        
                        for (int iii = 0; iii < int(single_constraints_list.size()); iii++) {
                            // get constraint function
                            single_constraint_struct cf = single_constraints_list[iii];
                            
                            //cout << cf.constraint_name << ", " << cf.flag << " --------------------------------" << endl;
                            
                            //Eigen::ArrayXd active_constraints = Eigen::ArrayXd::Zero(cf.length);
                            
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
                                
                                Eigen::VectorXd current_weights = weights.segment(ii * num_constraints + running_constraint_counter, cf.length);
                                /*
                                if (i == 0) {
                                    double constraint_gradient_norm = std::sqrt(single_dg_x.norm() * single_dg_x.norm() + single_dg_u.norm() * single_dg_u.norm());
                                    if (constraint_gradient_norm != 0) {
                                        constraint_scale_factor.segment(ii * num_constraints + running_constraint_counter, cf.length) = constraint_gradient_norm * Eigen::VectorXd::Ones(cf.length);
                                    }
                                    //std::cout << "Scale factor for " << cf.constraint_name << ": " << constraint_gradient_norm << std::endl;
                                }*/
                                
                                if (i == 0) {
                                    G_scale_factor = G_scale_factor + single_dg_x.norm() * single_dg_x.norm() + single_dg_u.norm() * single_dg_u.norm();
                                }
                                
                                // scale constraints down by scale factor
                                single_g = single_g / constraint_scale_factor[ii * num_constraints + running_constraint_counter];
                                single_dg_x = single_dg_x / constraint_scale_factor[ii * num_constraints + running_constraint_counter];
                                single_dg_u = single_dg_u / constraint_scale_factor[ii * num_constraints + running_constraint_counter];
                                
                                // new weight update?
                                single_g = single_g.cwiseProduct(current_weights);
                                single_dg_x = current_weights.asDiagonal() * single_dg_x;
                                single_dg_u = current_weights.asDiagonal() * single_dg_u;
                                
                                //cout << "cf flag is " << (cf.flag) << endl;
                                if (cf.flag == INEQUALITY) {
                                    for (int iiii = 0; iiii < cf.length; iiii++) {
                                        if (single_g[iiii] <= 0) {
                                            single_g[iiii] = 0;
                                            //cout << "in loop at " << iiii << endl;
                                            single_dg_x.block(iiii, 0, 1, num_states) = 0 * single_dg_x.block(iiii, 0, 1, num_states);
                                            single_dg_u.block(iiii, 0, 1, num_inputs) = 0 * single_dg_u.block(iiii, 0, 1, num_inputs);
                                        }
                                    }
                                }
                                
                                // for each constraint, check if constraint is active and update weights in that case
                                
                                for (int iv = 0; iv < cf.length; iv++) {
                                    // index of constraint
                                    int index = ii * num_constraints + running_constraint_counter + iv;
                                    // if constraint is violated, increase weight
                                    if (((cf.flag == INEQUALITY) & (single_g[iv] > 0)) || ((cf.flag == EQUALITY) & (single_g[iv] != 0))) {
                                        weights[index] = min(weights[index] * rho3_increase_rate, 1e9/rho3);
                                    }
                                }
                                
                                //cout << "single g after multiply: " << single_g.transpose().format(CleanFmt) << endl;
                                //cout << "SINGLE G NORM: " << single_g.lpNorm<Eigen::Infinity>() << endl;
                                //cout << "G NORM: " << g.lpNorm<Eigen::Infinity>() << endl;
                                
                                //cout << "Placing constraints in index [" << ii * num_constraints + running_constraint_counter << ", " << ii * num_constraints + running_constraint_counter + cf.length - 1 << "]" << endl;
                                g.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g;
                                placeinG(&tripletsG, single_dg_x, single_dg_u, ii, running_constraint_counter, cf.length);
                            }
                            
                            running_constraint_counter = running_constraint_counter + cf.length;
                        }
                        
                        for (int iii = 0; iii < int(double_constraints_list.size()); iii++) {
                            // get constraint function
                            double_constraint_struct cf = double_constraints_list[iii];
                            
                            //cout << cf.constraint_name << " --------------------------------" << endl;
                            
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
                                /*
                                if (i == 0) {
                                    double constraint_gradient_norm = std::sqrt(single_dg_x1.norm() * single_dg_x1.norm() + single_dg_u1.norm() * single_dg_u1.norm() + single_dg_x2.norm() * single_dg_x2.norm() + single_dg_u2.norm() * single_dg_u2.norm());
                                    if (constraint_gradient_norm != 0) {
                                        constraint_scale_factor.segment(ii * num_constraints + running_constraint_counter, cf.length) = constraint_gradient_norm * Eigen::VectorXd::Ones(cf.length);
                                    }
                                    //std::cout << "Scale factor for " << cf.constraint_name << ": " << constraint_gradient_norm << std::endl;
                                }*/
                                
                                if (i == 0) {
                                    G_scale_factor = G_scale_factor + single_dg_x1.norm() * single_dg_x1.norm() + single_dg_u1.norm() * single_dg_u1.norm() + single_dg_x2.norm() * single_dg_x2.norm() + single_dg_u2.norm() * single_dg_u2.norm();
                                }
                                
                                // scale constraints down by scale factor
                                single_g = single_g / constraint_scale_factor[ii * num_constraints + running_constraint_counter];
                                single_dg_x1 = single_dg_x1 / constraint_scale_factor[ii * num_constraints + running_constraint_counter];
                                single_dg_u1 = single_dg_u1 / constraint_scale_factor[ii * num_constraints + running_constraint_counter];
                                single_dg_x2 = single_dg_x2 / constraint_scale_factor[ii * num_constraints + running_constraint_counter];
                                single_dg_u2 = single_dg_u2 / constraint_scale_factor[ii * num_constraints + running_constraint_counter];
                                
                                Eigen::VectorXd current_weights = weights.segment(ii * num_constraints + running_constraint_counter, cf.length);
                                //cout << "current_weights: " << current_weights.transpose().format(CleanFmt) << endl;
                                //cout << "single g before multiply: " << single_g.transpose().format(CleanFmt) << endl;
                                
                                // new weight update?
                                single_g = single_g.cwiseProduct(current_weights);
                                single_dg_x1 = current_weights.asDiagonal() * single_dg_x1;
                                single_dg_u1 = current_weights.asDiagonal() * single_dg_u1;
                                single_dg_x2 = current_weights.asDiagonal() * single_dg_x2;
                                single_dg_u2 = current_weights.asDiagonal() * single_dg_u2;
                                
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
                                
                                //cout << "single g after multiply: " << single_g.transpose().format(CleanFmt) << endl;
                                //cout << "SINGLE G NORM: " << single_g.lpNorm<Eigen::Infinity>() << endl;
                                //cout << "G NORM: " << g.lpNorm<Eigen::Infinity>() << endl;
                                
                                // for each constraint, check if constraint is active and update weights in that case
                                
                                for (int iv = 0; iv < cf.length; iv++) {
                                    // index of constraint
                                    int index = ii * num_constraints + running_constraint_counter + iv;
                                    // if constraint is violated, increase weight
                                    if (((cf.flag == INEQUALITY) & (single_g[iv] > 0)) || ((cf.flag == EQUALITY) & (single_g[iv] != 0))) {
                                        weights[index] = min(weights[index] * rho3_increase_rate, 1e9/rho3);
                                    }
                                }
                                
                                //cout << "Placing constraints in index [" << ii * num_constraints + running_constraint_counter << ", " << ii * num_constraints + running_constraint_counter + cf.length - 1 << "]" << endl;
                                g.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g;
                                placeinG(&tripletsG, single_dg_x1, single_dg_u1, ii, running_constraint_counter, cf.length);
                                placeinG(&tripletsG, single_dg_x2, single_dg_u2, ii+1, running_constraint_counter, cf.length);
                                
                            }
                            
                            running_constraint_counter = running_constraint_counter + cf.length;
                        }
                        
                        // iterate through all constraints and print weights
                        
                        G.setFromTriplets(tripletsG.begin(), tripletsG.end());
                        
                        h = G * y - g;
                        
                        // scale M and c (once) and save scale factor
                        if (i == 0) {
                            if (SCALING) {
                                G_scale_factor = std::sqrt(G_scale_factor);
                            } else {
                                G_scale_factor = 1;
                            }
                            std::cout << "G scale factor: " << G.norm() << std::endl;
                        }
                        if (G_scale_factor > 0) {
                            G = G / G_scale_factor;
                            h = h / G_scale_factor;
                        }
                        
                        // first ADMM update
                        admm_update1_time_start = std::chrono::system_clock::now(); // start timer
                        
                        x = proximalUpdateObjective(y, lambda, R, b, rho1);
                        
                        admm_update1_time_end = std::chrono::system_clock::now(); // end timer
                        admm_update1_timer = admm_update1_timer + (duration_cast<duration<double>>(admm_update1_time_end - admm_update1_time_start)).count();
                        
                        // second ADMM update
                        admm_update2_time_start = std::chrono::system_clock::now(); // start timer
                        
                        y = proximalUpdateConstraints(x, lambda, M, c, G, h, rho1, rho2, rho3);
                        //std::cout << "Change from objective update: " << (y - x).lpNorm<2>() << std::endl;
                        //std::cout << "Change from oprevious traj: " << (y - y_prev).lpNorm<2>() << std::endl;
                        y_prev = y;
                        
                        admm_update2_time_end = std::chrono::system_clock::now(); // end timer
                        admm_update2_timer = admm_update2_timer + (duration_cast<duration<double>>(admm_update2_time_end - admm_update2_time_start)).count();
                        
                        // dual update
                        lambda = lambda + rho1 * (x - y);
                        
                        // calculate statistics
                        scaledObjective = y.transpose() * R * y;
                        unscaledObjective = scaledObjective * R_scale_factor;
                        
                        scaledFeasibilityVector = M * y - c;
                        scaledFeasibilityNorm = scaledFeasibilityVector.lpNorm<Eigen::Infinity>();
                        
                        unscaledConstraintVector = G_scale_factor * g.cwiseProduct(constraint_scale_factor.cwiseQuotient(weights)); //previous: constraintVector = G * y - h;
                        unscaledConstraintNorm = unscaledConstraintVector.lpNorm<Eigen::Infinity>();
                        
                        fullObjective = scaledObjective + rho2 * pow(scaledFeasibilityVector.lpNorm<2>(), 2) + rho3 * pow(g.lpNorm<2>(), 2);
                        
                        // calculate flags
                        not_feasible = (scaledFeasibilityVector.lpNorm<Eigen::Infinity>() * M_scale_factor > tol_feasibility || unscaledConstraintVector.lpNorm<Eigen::Infinity>() > tol_constraints);
                        not_converged = (abs(oldObjective - unscaledObjective)/unscaledObjective > tol_objective);
                        
                        // increase rho2
                        rho2 = min(rho2 * rho2_increase_rate, rho_max);
                        rho3 = min(rho3 * rho3_increase_rate, rho_max);
                        
                        // decrease rho1 if constraints are mostly satisfied
                        if (i > 100) {
                            rho1 = max(rho1_min, rho1 / rho1_decrease_rate);
                        }
                        
                        // print / compute info
                        if (i % 1 == 0) {
                            output_stream << "Iteration " << i << " -- objective cost: " << unscaledObjective << " -- feasibility: " << scaledFeasibilityNorm << " -- unscaled feasibility: " << scaledFeasibilityNorm * M_scale_factor << " -- constraints: " << g.lpNorm<Eigen::Infinity>() << " -- unscaled constraints: " << unscaledConstraintNorm << "-- full objective: " << fullObjective << "\n";
                        }
                        
                        tripletsM.clear();
                        tripletsG.clear();
                        oldFeasibilityNorm = scaledFeasibilityNorm;
                        oldObjective = unscaledObjective;
                        
                        i++;
                    }
                    
                    // print final convergence output
                    output_stream << "Finished at iteration " << i << " -- objective cost: " << unscaledObjective << " -- feasibility: " << scaledFeasibilityNorm << " -- unscaled feasibility: " << scaledFeasibilityNorm * M_scale_factor << " -- constraints: " << g.lpNorm<Eigen::Infinity>() << " -- unscaled constraints: " << unscaledConstraintNorm << "-- full objective: " << fullObjective << "\n";
                    
                    // fill in solution
                    for (int ii = 0; ii < N; ii++) {
                        solution_x.block(0, ii, num_states, 1) = y.segment(ii * num_states, num_states);
                        solution_u.block(0, ii, num_inputs, 1) = y.segment(N * num_states + ii * num_inputs, num_inputs);
                    }
                    
                    //cout << "\n\n";
                    //cout << solution_x.transpose() << "\n\n";
                    
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
                    
                    // save number of iterations
                    num_latest_iterations = i;
                    
                    // return
                    if (i == max_iter) {
                        return "IterationLimit";
                    } else {
                        return "SolutionFound";
                    }
                }
                
                /*
                Eigen::VectorXd AdmmSolverScaled::proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> y, Eigen::Ref<Eigen::VectorXd> lambda, Eigen::Ref<Eigen::MatrixXd> R, Eigen::Ref<Eigen::VectorXd> b, double rho1) {
                    Eigen::MatrixXd temp = 2 * R + rho1 * Eigen::MatrixXd::Identity(N * (num_states + num_inputs), N * (num_states + num_inputs));
                    return temp.llt().solve(rho1 * y - lambda - b);
                } */
                
                /*
                 Eigen::VectorXd AdmmSolverScaled::proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> y, Eigen::Ref<Eigen::VectorXd> lambda, Eigen::Ref<Eigen::MatrixXd> R, double rho1) {
                 Eigen::MatrixXd temp = 2 * R + rho1 * Eigen::MatrixXd::Identity(N * (num_states + num_inputs), N * (num_states + num_inputs));
                 return temp.llt().solve(rho1 * (y - lambda) - b);
                 }
                */
                
                // SECOND PROXIMAL UPDATE
                Eigen::VectorXd AdmmSolverScaled::proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> lambda, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3) {
                    Eigen::MatrixXd Mt = M.transpose();
                    Eigen::MatrixXd Gt = G.transpose();
                    
                    Eigen::MatrixXd temp = rho1 * Eigen::MatrixXd::Identity(M.cols(), M.cols()) + rho2 * Mt * M + rho3 * Gt * G;
                    return temp.llt().solve(lambda + rho1 * x + rho2 * Mt * c + rho3 * Gt * h);
                }
                
                /*
                 Eigen::VectorXd AdmmSolverScaled::proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> lambda, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3) {
                 Eigen::MatrixXd Mt = M.transpose();
                 Eigen::MatrixXd Gt = G.transpose();
                 
                 Eigen::MatrixXd temp = rho1 * Eigen::MatrixXd::Identity(M.cols(), M.cols()) + 2 * rho2 * Mt * M + 2 * rho3 * Gt * G;
                 return temp.llt().solve(rho1 * (lambda + x) + 2 * rho2 * Mt * c + 2 * rho3 * Gt * h);
                 }
                */
                 
                 
            } // namespace admm_solver
        } // namespace trajectory_optimization
    } // namespace systems
} // namespace drake

