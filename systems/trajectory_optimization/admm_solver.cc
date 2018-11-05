// Version with obstacle constraints(and sparse M and G)!

// run as: g++ -I /usr/local/include/eigen3/ -O2 solveADMMconstraints.cpp

#include "drake/systems/trajectory_optimization/admm_solver.h"

using namespace std::chrono;
using namespace std::placeholders;

namespace drake {
    namespace systems {
        namespace trajectory_optimization {
            namespace admm_solver {
            
            /* ---------------------------------------------- SOLVER INITIALIZATION ---------------------------------------------- */
            
            AdmmSolver::AdmmSolver(systems::System<double>* par_system, Eigen::VectorXd par_x0, Eigen::VectorXd par_xf, double par_T, int par_N, int par_max_iter) {
                
                system = par_system;
                context = system->CreateDefaultContext();
                input_port_value = &context->FixInputPort(0, system->AllocateInputVector(system->get_input_port(0)));
                
                // create an autodiff version of the system and context
                autodiff_system = drake::systems::System<double>::ToAutoDiffXd(*system);
                autodiff_context = autodiff_system->CreateDefaultContext();
                autodiff_context->SetTimeStateAndParametersFrom(*context);
                
                initial_rho1 = 500;
                initial_rho2 = 10000;
                initial_rho3 = 1000;
                
                rho1_decrease_rate = 1.02; // decrease by a constant factor if constraints are satisfied
                rho2_increase_rate = 1.02; // increases by a constant factor with every iteration
                rho3_increase_rate = 1.02; // increases by a constant factor with every iteration
                rho_max = 1e9; // maximum value for rho2 and rho3
                rho1_min = 100;
                
                num_states = context->get_num_total_states();
                num_inputs = system->get_input_port(0).size();
                
                costQ = Eigen::MatrixXd::Zero(num_states, num_states);
                costR = Eigen::MatrixXd::Identity(num_inputs, num_inputs);
                
                x0 = Eigen::VectorXd(par_x0);
                xf = Eigen::VectorXd(par_xf);
                
                num_constraints = 0;
                
                T = par_T;
                N = par_N;
                dt = T/N;
                
                tol_feasibility = 1e-4;
                tol_constraints = 1e-4;
                tol_objective = 1e-4;
                max_iter = par_max_iter;
                
                // bounds on state and input
                x_lower_bound = Eigen::VectorXd::Zero(num_states);
                x_upper_bound = Eigen::VectorXd::Zero(num_states);
                u_lower_bound = Eigen::VectorXd::Zero(num_inputs);
                u_upper_bound = Eigen::VectorXd::Zero(num_inputs);
                
                solve_flag = false;
            }
            
            void AdmmSolver::setRho1(double rho) {
                initial_rho1 = rho;
            }
            
            void AdmmSolver::setRho2(double rho) {
                initial_rho2 = rho;
            }
            
            void AdmmSolver::setRho3(double rho) {
                initial_rho3 = rho;
            }
            
            void AdmmSolver::setFeasibilityTolerance(double tol) {
                tol_feasibility = tol;
            }
            
            void AdmmSolver::setConstraintTolerance(double tol) {
                tol_constraints = tol;
            }
            
            void AdmmSolver::setStateUpperBound(Eigen::Ref<Eigen::VectorXd> bound) {
                if (bound.size() != num_states) {
                    cerr << "State upper bound must have length equal to the number of states.\n";
                }
                x_upper_bound = bound;
                
                // make constraint (uses lambda to bind "this" object to the member function)
                single_constraint_function f = [this](double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u){stateUpperBoundConstraint(t, x, u, g, dg_x, dg_u);};
                struct single_constraint_struct cf = {f, INEQUALITY, num_states, "stateUpperBound"};
                
                // add constraint to list
                single_constraints_list.push_back(cf);
                
                // increment overall constraint counter
                num_constraints = num_constraints + num_states;
            }
            
            void AdmmSolver::setStateLowerBound(Eigen::Ref<Eigen::VectorXd> bound) {
                if (bound.size() != num_states) {
                    cerr << "State lower bound must have length equal to the number of states.\n";
                }
                x_lower_bound = bound;
                
                // make constraint
                single_constraint_function f = [this](double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u){stateLowerBoundConstraint(t, x, u, g, dg_x, dg_u);};
                struct single_constraint_struct cf = {f, INEQUALITY, num_states, "stateLowerBound"};
                
                // add constraint to list
                single_constraints_list.push_back(cf);
                
                // increment overall constraint counter
                num_constraints = num_constraints + num_states;
            }
            
            void AdmmSolver::setInputUpperBound(Eigen::Ref<Eigen::VectorXd> bound) {
                if (bound.size() != num_inputs) {
                    cerr << "Input upper bound must have length equal to the number of inputs.\n";
                }
                u_upper_bound = bound;
                
                // make constraint
                single_constraint_function f = [this](double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u){inputUpperBoundConstraint(t, x, u, g, dg_x, dg_u);};
                struct single_constraint_struct cf = {f, INEQUALITY, num_inputs, "inputUpperBound"};
                
                // add constraint to list
                single_constraints_list.push_back(cf);
                
                // increment overall constraint counter
                num_constraints = num_constraints + num_inputs;
            }
            
            void AdmmSolver::setInputLowerBound(Eigen::Ref<Eigen::VectorXd> bound) {
                if (bound.size() != num_inputs) {
                    cerr << "Input lower bound must have length equal to the number of inputs.\n";
                }
                u_lower_bound = bound;
                
                // make constraint
                single_constraint_function f = [this](double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u){inputLowerBoundConstraint(t, x, u, g, dg_x, dg_u);};
                struct single_constraint_struct cf = {f, INEQUALITY, num_inputs, "inputLowerBound"};
                
                // add constraint to list
                single_constraints_list.push_back(cf);
                
                // increment overall constraint counter
                num_constraints = num_constraints + num_inputs;
            }
            
            Eigen::MatrixXd AdmmSolver::getSolutionStateTrajectory() {
                return solution_x;
            }
            
            Eigen::MatrixXd AdmmSolver::getSolutionInputTrajectory() {
                return solution_u;
            }
            
            trajectories::PiecewisePolynomial<double> AdmmSolver::reconstructStateTrajectory() {
                // check if problem has been solved
                if (!solve_flag) {
                    cerr << "Cannot reconstruct trajectory -- problem has not been solved successfully.\n";
                }
                
                std::vector<double> times_vec(N);
                std::vector<Eigen::MatrixXd> states(N);
                std::vector<Eigen::MatrixXd> derivatives(N);
                std::unique_ptr<ContinuousState<double> > continuous_state(system->AllocateTimeDerivatives());
                
                for (int i = 0; i < N; i++) {
                    times_vec[i] = dt * i;
                    states[i] = solution_x.block(0, i, num_states, 1);
                    if (context->get_num_input_ports() > 0) {
                        input_port_value->GetMutableVectorData<double>()->SetFromVector(solution_u.block(0, i, num_inputs, 1));
                    }
                    context->get_mutable_continuous_state().SetFromVector(states[i]);
                    system->CalcTimeDerivatives(*context, continuous_state.get());
                    derivatives[i] = continuous_state->CopyToVector();
                }
                return trajectories::PiecewisePolynomial<double>::Cubic(times_vec, states, derivatives);
            }
            
            
            trajectories::PiecewisePolynomial<double> AdmmSolver::reconstructInputTrajectory() {
                // check if problem has been solved
                if (!solve_flag) {
                    cerr << "Cannot reconstruct trajectory -- problem has not been solved successfully.\n";
                }
                DRAKE_DEMAND(context->get_num_input_ports() > 0);
                std::vector<double> times_vec(N);
                std::vector<Eigen::MatrixXd> inputs(N);
                
                for (int i = 0; i < N; i++) {
                    times_vec[i] = dt * i;
                    inputs[i] = solution_u.block(0, i, num_inputs, 1);
                }
                return trajectories::PiecewisePolynomial<double>::FirstOrderHold(times_vec, inputs);
            }
            
            void AdmmSolver::addInequalityConstraintToAllKnotPoints(single_constraint_function f, int constraint_size, std::string constraint_name) {
                // make individual constraint structure
                struct single_constraint_struct cf = {f, INEQUALITY, constraint_size, constraint_name};
                
                // put onto overall constraint list
                single_constraints_list.push_back(cf);
                num_constraints = num_constraints + constraint_size;
            }
            
            void AdmmSolver::addEqualityConstraintToAllKnotPoints(single_constraint_function f, int constraint_size, std::string constraint_name) {
                // make individual constraint structure
                struct single_constraint_struct cf = {f, EQUALITY, constraint_size, constraint_name};
                
                // put onto overall constraint list
                single_constraints_list.push_back(cf);
                num_constraints = num_constraints + constraint_size;
            }
            
            void AdmmSolver::addInequalityConstraintToConsecutiveKnotPoints(double_constraint_function f, int constraint_size, std::string constraint_name) {
                // make individual constraint structure
                struct double_constraint_struct cf = {f, INEQUALITY, constraint_size, constraint_name};
                
                // put onto overall constraint list
                double_constraints_list.push_back(cf);
                num_constraints = num_constraints + constraint_size;
            }
            
            void AdmmSolver::addEqualityConstraintToConsecutiveKnotPoints(double_constraint_function f, int constraint_size, std::string constraint_name) {
                // make individual constraint structure
                struct double_constraint_struct cf = {f, EQUALITY, constraint_size, constraint_name};
                
                // put onto overall constraint list
                double_constraints_list.push_back(cf);
                num_constraints = num_constraints + constraint_size;
            }

            
            
            /* ---------------------------------------------- SOLVE METHOD ---------------------------------------------- */
            
            void AdmmSolver::solve(Eigen::Ref<Eigen::VectorXd> y) {
                
                std::cout << "num states (in ADMM): " << num_states << endl;
                std::cout << "num inputs (in ADMM): " << num_inputs << endl;
                
                /* --- allocate memory --- */
                std::cout << "Before solve starts...\n";
                std::cout << "num constraints per point = " << num_constraints << "\n";
                std::cout << "constraints list length = " << single_constraints_list.size() << "\n";
                
                // allocate array for x, y, lambda
                Eigen::VectorXd x(y);
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
                
                // allocate memory for midpoint values
                Eigen::VectorXd mid_state(num_states);
                Eigen::VectorXd mid_input(num_inputs);
                Eigen::VectorXd temp;
                Eigen::VectorXd feasibilityVector = Eigen::VectorXd::Zero((N+1) * num_states);
                Eigen::VectorXd constraintVector = Eigen::VectorXd::Zero((N+1) * num_states);
                double objective;
                double oldObjective;
                double feasibilityNorm;
                double oldFeasibilityNorm;
                double constraintNorm;
                double full_objective;
                
                // cost matrix
                Eigen::MatrixXd R = Eigen::MatrixXd::Zero(N * (num_states + num_inputs), N * (num_states + num_inputs));
                R.block(N * num_states, N * num_states, N * num_inputs, N * num_inputs) = Eigen::MatrixXd::Identity(N * num_inputs, N * num_inputs);
                
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
                ofstream output_G; // for debugging
                
                // set rho's to initial values
                double rho1 = initial_rho1;
                double rho2 = initial_rho2;
                double rho3 = initial_rho3;
                
                // open output file for writing y
                if (DEBUG) {
                    output_file.open("/Users/ira/Documents/drake/examples/quadrotor/output/accel/single_run_admm_y.txt");
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening output file.";
                        return;
                    }
                }
                std:: cout << "hi" << endl;
                
                int i = 0;
                while (i < max_iter && (i < 1 || feasibilityVector.lpNorm<Eigen::Infinity>() > tol_feasibility || constraintVector.lpNorm<Eigen::Infinity>() > tol_constraints)) {
                    
                    // write y to file
                    if (DEBUG) {
                        output_file << y.transpose() << endl;
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
                        
                        /*
                        if (ii == N - 2) {
                            std::cout << "f:\n";
                            for (int iii = 0; iii < N-1; iii++) {
                                std::cout << f[iii] << endl;
                            }
                            std::cout << "A:\n";
                            for (int iii = 0; iii < N-1; iii++) {
                                std::cout << A[iii] << endl;
                            }
                            std::cout << "B:\n";
                            for (int iii = 0; iii < N-1; iii++) {
                                std::cout << B[iii] << endl;
                            }
                        } */
                        
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
                    //std::cout << "M:\n" << M << endl;
                    
                    /*
                    for (int ii = 0; ii < N; ii++) {
                        // get current state/input
                        Eigen::VectorXd state = y.segment(ii * num_states, num_states);
                        Eigen::VectorXd input = y.segment(N * num_states + ii * num_inputs, num_inputs);
                        //std::cout << input << ", ";
                        
                        int running_constraint_counter = 0;
                        
                        for (int iii = 0; iii < int(single_constraints_list.size()); iii++) {
                            // get constraint function
                            single_constraint_struct cf = single_constraints_list.at(iii);
                            
                            // prepare matrices (OPTIMIZE LATER)
                            Eigen::VectorXd single_g = Eigen::VectorXd::Zero(cf.length);
                            Eigen::MatrixXd single_dg_x = Eigen::MatrixXd::Zero(cf.length, num_states);
                            Eigen::MatrixXd single_dg_u = Eigen::MatrixXd::Zero(cf.length, num_inputs);
                            
                            // evaluate constraints
                            cf.function(ii, state, input, single_g, single_dg_x, single_dg_u);
                            
                            if (cf.flag == INEQUALITY) {
                                for (int iiii = 0; iiii < cf.length; iiii++) {
                                    if (single_g[iiii] <= 0) {
                                        single_g[iiii] = 0;
                                        single_dg_x.block(iiii, 0, 1, num_states) = 0 * single_dg_x.block(iiii, 0, 1, num_states);
                                        single_dg_u.block(iiii, 0, 1, num_inputs) = 0 * single_dg_u.block(iiii, 0, 1, num_inputs);
                                    }
                                }
                            }
                            
                            g.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g;
                            placeinG(&tripletsG, single_dg_x, single_dg_u, ii, running_constraint_counter, cf.length);
                            
                            running_constraint_counter = running_constraint_counter + cf.length;
                        }
                    } */
                    
                    int running_constraint_counter = 0;
                    
                    for (int iii = 0; iii < int(single_constraints_list.size()); iii++) {
                        // get constraint function
                        single_constraint_struct cf = single_constraints_list.at(iii);
                        
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
                            
                            if (cf.flag == INEQUALITY) {
                                for (int iiii = 0; iiii < cf.length; iiii++) {
                                    if (single_g[iiii] <= 0) {
                                        single_g[iiii] = 0;
                                        single_dg_x.block(iiii, 0, 1, num_states) = 0 * single_dg_x.block(iiii, 0, 1, num_states);
                                        single_dg_u.block(iiii, 0, 1, num_inputs) = 0 * single_dg_u.block(iiii, 0, 1, num_inputs);
                                    } else {
                                        //std::cout << "Point " << ii << " violates subconstraint " << iiii << " of " << cf.constraint_name << ".\n";
                                        //std::cout << "State: " << state << endl;
                                    }
                                }
                            }
                            
                            g.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g;
                            placeinG(&tripletsG, single_dg_x, single_dg_u, ii, running_constraint_counter, cf.length);
                        }
                        running_constraint_counter = running_constraint_counter + cf.length;
                    }
                    
                    for (int iii = 0; iii < int(double_constraints_list.size()); iii++) {
                        // get constraint function
                        double_constraint_struct cf = double_constraints_list.at(iii);
                        
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
                            
                            //std::cout << "single dg x1:\n" << single_dg_x1;
                            //std::cout << "single dg u1:\n" << single_dg_u1;
                            //std::cout << "single dg x2:\n" << single_dg_x2;
                            //std::cout << "single dg u2:\n" << single_dg_u2;
                            
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
                            
                            g.segment(ii * num_constraints + running_constraint_counter, cf.length) = single_g;
                            placeinG(&tripletsG, single_dg_x1, single_dg_u1, ii, running_constraint_counter, cf.length);
                            placeinG(&tripletsG, single_dg_x2, single_dg_u2, ii+1, running_constraint_counter, cf.length);
                            
                        }
                        running_constraint_counter = running_constraint_counter + cf.length;
                    }
                     
                    G.setFromTriplets(tripletsG.begin(), tripletsG.end());
                    
                    //std::cout << G << endl;
                    
                    // print to output file
                    /*
                    if (i == 0) {
                        output_G.open("/Users/ira/Documents/drake/examples/quadrotor/output/G.txt");
                        if (!output_G.is_open()) {
                            std::cerr << "Problem opening G output file." << endl;
                        } else {
                            output_G << G;
                        }
                        output_G.close();
                    } */
                    
                    //std::cout << "\nG rows and cols:\n" << G.rows() << " " << G.cols() << "\n";
                    //std::cout << "\nG block:\n" << G.block(num_constraints, num_states, num_constraints, 2 * num_states) << "\n";
                    //std::cout << "g:\n" << g << "\n";
                    
                    h = G * y - g;
                    
                    // first ADMM update
                    admm_update1_time_start = std::chrono::system_clock::now(); // start timer
                    
                    temp = y - lambda;
                    x = proximalUpdateObjective(temp, R, rho1);
                    
                    admm_update1_time_end = std::chrono::system_clock::now(); // end timer
                    admm_update1_timer = admm_update1_timer + (duration_cast<duration<double>>(admm_update1_time_end - admm_update1_time_start)).count();
                    
                    // second ADMM update
                    admm_update2_time_start = std::chrono::system_clock::now(); // start timer
                    
                    temp = x + lambda;
                    y = proximalUpdateConstraints(temp, M, c, G, h, rho1, rho2, rho3);
                    
                    admm_update2_time_end = std::chrono::system_clock::now(); // end timer
                    admm_update2_timer = admm_update2_timer + (duration_cast<duration<double>>(admm_update2_time_end - admm_update2_time_start)).count();
                    
                    // dual update
                    lambda = lambda + x - y;
                    
                    // increase rho2 and rho3
                    objective = y.transpose() * R * y;
                    feasibilityVector = M * y - c;
                    feasibilityNorm = feasibilityVector.lpNorm<Eigen::Infinity>();
                    constraintVector = g; //previous: constraintVector = G * y - h;
                    constraintNorm = constraintVector.lpNorm<Eigen::Infinity>();
                    full_objective = objective + rho2 * pow(feasibilityVector.lpNorm<2>(), 2) + rho3 * pow(constraintVector.lpNorm<2>(), 2);
                    
                    // increase rho2
                    rho2 = min(rho2 * rho2_increase_rate, rho_max);
                    //rho3 = min(rho3 * rho3_increase_rate, rho_max);
                    
                    // decrease rho1 if constraints are mostly satisfied
                    if (i > 2 && (feasibilityNorm - oldFeasibilityNorm < 0.001)) {
                        rho1 = max(rho1_min, rho1 / rho1_decrease_rate);
                    }
                    
                    // print / compute info
                    if (i % 20 == 0) {
                        cout << "Iteration " << i << " -- objective cost: " << objective << " -- feasibility (inf-norm): " << feasibilityNorm << " -- constraint satisfaction (inf-norm): " << constraintNorm << "-- full objective: " << full_objective << "\n";
                    }
                    
                    tripletsM.clear();
                    tripletsG.clear();
                    oldFeasibilityNorm = feasibilityNorm;
                    oldObjective = objective;
                    i++;
                }
                
                // print final convergence output
                cout << "Finished at iteration " << i << " -- objective cost: " << objective << " -- feasibility (inf-norm): " << feasibilityNorm << " -- constraint satisfaction (inf-norm): " << constraintNorm << "\n";
                
                // fill in solution
                for (int ii = 0; ii < N; ii++) {
                    solution_x.block(0, ii, num_states, 1) = y.segment(ii * num_states, num_states);
                    solution_u.block(0, ii, num_inputs, 1) = y.segment(N * num_states + ii * num_inputs, num_inputs);
                }
                
                //cout << "\n\n";
                //cout << solution_x.transpose() << "\n\n";
                
                // print timing information
                cout << "\n---------------------------------\n";
                cout << "TIMING:\n";
                cout << "---------------------------------\n";
                cout << "Total time spent in dynamics: " << admm_dynamics_timer << " (sec)\n";
                cout << "Total time spent in first ADMM update: " << admm_update1_timer << " (sec)\n";
                cout << "Total time spent in second ADMM update: " << admm_update2_timer << " (sec)\n";
                cout << "Total update time: " << admm_update1_timer + admm_update2_timer << " (sec)\n";
                solve_flag = true;
                output_file.close();
            }
            
            
            
            /* ---------------------------------------------- FUNCTIONS ---------------------------------------------- */
            
            // really naive implementation... whatever
            /*
            int AdmmSolver::nonzeroCount(Eigen::Ref<Eigen::MatrixXd> A) {
                int count = 0;
                for (int i = 0; i < A.rows(); i++) {
                    for (int ii = 0; ii < A.cols(); ii++) {
                        if (A(i, ii) != 0) {
                            count++;
                        }
                    }
                }
                return count;
            } */
            
            
            // FIRST PROXIMAL UPDATE
            Eigen::VectorXd AdmmSolver::proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> nu, Eigen::Ref<Eigen::MatrixXd> R, double rho1) {
                Eigen::MatrixXd temp = 2 * R + rho1 * Eigen::MatrixXd::Identity(N * (num_states + num_inputs), N * (num_states + num_inputs));
                return temp.llt().solve(rho1 * nu);
            }
            
            
            // SECOND PROXIMAL UPDATE
            Eigen::VectorXd AdmmSolver::proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> nu, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3) {
                Eigen::MatrixXd Mt = M.transpose();
                Eigen::MatrixXd Gt = G.transpose();
                
                Eigen::MatrixXd temp = 2 * rho2 * Mt * M + 2 * rho3 * Gt * G + rho1 * Eigen::MatrixXd::Identity(M.cols(), M.cols());
                //std::cout << "Size of matrix being inverted: " << temp.size() << endl;
                return temp.llt().solve(2 * rho2 * Mt * c + 2 * rho3 * Gt * h + rho1 * nu);
            }
            
            double AdmmSolver::getStateFromY(Eigen::Ref<Eigen::VectorXd> y, int time_index, int index) {
                return (y[time_index * num_states + index]);
            }
            
            double AdmmSolver::getInputFromY(Eigen::Ref<Eigen::VectorXd> y, int time_index, int index) {
                return (y[N * num_states + time_index * num_inputs + index]);
            }
            
            
            // DOUBLE INTEGRATOR DYNAMICS
            void AdmmSolver::integratorDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> state, Eigen::Ref<Eigen::VectorXd> input, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii) {
                f[0] = state[1];
                f[1] = input[0];
                
                Aii(0, 0) = 0; Aii(0, 1) = 1;
                Aii(1, 0) = 0; Aii(1, 1) = 0;
                
                Bii(0, 0) = 0; Bii(1, 0) = 1;
            }
            
            /*
            // QUAD DYNAMICS
            void AdmmSolver::quadDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii) {
                
                double quad_L = 0.25; // length of rotor arm
                double quad_mass = 0.486; // mass of quadrotor
                double quad_I = 0.00383; // moment of inertia
                double g = 9.81;
                
                f[0] = x[3];
                f[1] = x[4];
                f[2] = x[5];
                f[3] = -sin(x[2])/quad_mass * (u[0] + u[1]);
                f[4] = -g + cos(x[2])/quad_mass * (u[0] + u[1]);
                f[5] = quad_L/quad_I * (-u[0] + u[1]);
                
                Aii(0, 3) = 1; Aii(1, 4) = 1; Aii(2, 5) = 1;
                Aii(3, 2) = -cos(x[2])/quad_mass * (u[0] + u[1]);
                Aii(4, 2) = -sin(x[2])/quad_mass * (u[0] + u[1]);
                
                Bii(3, 0) = -sin(x[2])/quad_mass;
                Bii(3, 1) = -sin(x[2])/quad_mass;
                Bii(4, 0) = cos(x[2])/quad_mass;
                Bii(4, 1) = cos(x[2])/quad_mass;
                Bii(5, 0) = -quad_L/quad_I;
                Bii(5, 1) = quad_L/quad_I;
            } */
            
            /*
             // ALL CONSTRAINTS? (FOR TESTING)
             void AdmmSolver::allConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg) {
             
             int num_states = 6; int num_inputs = 2;
             int N = (x.size() / num_states);
             
             // useful
             int nn = num_inputs + num_states;
             int num_constraints = 3 * 6 + 2 * (6 + 2);
             
             // define state and input constraint values
             Eigen::VectorXd x_upper_bound(6); x_upper_bound << 6.0, 7.0, 10000, 10000, 10000, 10000;
             Eigen::VectorXd x_lower_bound(6); x_lower_bound << -1.0, 0.0, -10000, -10000, -10000, -10000;
             Eigen::VectorXd u_upper_bound(2); u_upper_bound << 10000, 10000;
             Eigen::VectorXd u_lower_bound(2); u_lower_bound << -10000, -10000;
             
             // define obstacles
             double num_obstacles = 6;
             struct obstacle obs[6] = {{{1.5, 0.5}, 0.5}, {{1.5, 2.5}, 0.5}, {{1.5, 4.5}, 0.5}, {{3.5, 1.5}, 0.5}, {{3.5, 3.5}, 0.5}, {{3.5, 5.5}, 0.5}};
             
             // call state and input bound constraints
             stateInputConstraints(time_index, x, u, x_upper_bound, x_lower_bound, u_upper_bound, u_lower_bound, num_states, num_inputs, g.segment(0, 2 * nn), dg.block(0, 0, 2 * nn, nn));
             
             // call obstacle constraints
             obstacleConstraints(time_index, x, u, obs, num_obstacles, N, num_states, num_inputs, g.segment(2 * nn, num_obstacles * 3), dg.block(2 * nn, 0, num_obstacles * 3, nn));
             }
             */
            
            // STATE INPUT CONSTRAINTS
            void AdmmSolver::stateUpperBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                g = x - x_upper_bound;
                dg_x = Eigen::MatrixXd::Identity(num_states, num_states);
                dg_u = Eigen::MatrixXd::Zero(num_states, num_inputs);
            }
            
            void AdmmSolver::stateLowerBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                g = x_lower_bound - x;
                dg_x = -1 * Eigen::MatrixXd::Identity(num_states, num_states);
                dg_u = Eigen::MatrixXd::Zero(num_states, num_inputs);
            }
            
            void AdmmSolver::inputUpperBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                g = u - u_upper_bound;
                dg_x = Eigen::MatrixXd::Zero(num_inputs, num_states);
                dg_u = Eigen::MatrixXd::Identity(num_inputs, num_inputs);
            }
            
            void AdmmSolver::inputLowerBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                g = u_lower_bound - u;
                dg_x = Eigen::MatrixXd::Zero(num_inputs, num_states);
                dg_u = -1 * Eigen::MatrixXd::Identity(num_inputs, num_inputs);
            }
            /*
            void AdmmSolver::stateInputConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> x_upper_bound, Eigen::Ref<Eigen::VectorXd> x_lower_bound, Eigen::Ref<Eigen::VectorXd> u_upper_bound, Eigen::Ref<Eigen::VectorXd> u_lower_bound, int num_states, int num_inputs, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg) {
                
                int nn = num_states + num_inputs;
                
                // entries of g
                g.segment(0, num_states) = x_lower_bound - x;
                g.segment(num_states, num_inputs) = u_lower_bound - u;
                g.segment(num_inputs + num_states, num_states) = x - x_upper_bound;
                g.segment(num_inputs + 2 * num_states, num_inputs) =  u - u_upper_bound;
                
                // entries of dg
                dg.block(0, 0, nn, nn) = -1 * Eigen::MatrixXd::Identity(nn, nn);
                dg.block(nn, 0, nn, nn) = Eigen::MatrixXd::Identity(nn, nn);
            } */
            
            /*
             // OBSTACLE CONSTRAINTS
             void AdmmSolver::obstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, obstacle* obs, int num_constraints, int N, int num_states, int num_inputs, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg) {
             
             double quad_L = 0.25;
             
             double right_side_x, right_side_y;
             double left_side_x, left_side_y;
             double rad_squared;
             
             for (int i = 0; i < num_constraints; i++) {
             // pre-compute
             rad_squared = obs[i].radius * obs[i].radius;
             right_side_x = x[0] + quad_L * cos(x[2]) - obs[i].center[0];
             right_side_y = x[1] + quad_L * sin(x[2]) - obs[i].center[1];
             left_side_x = x[0] - quad_L * cos(x[2]) - obs[i].center[0];
             left_side_y = x[1] - quad_L * sin(x[2]) - obs[i].center[1];
             
             // entries of d
             g(3 * i) = rad_squared - (obs[i].center[0] - x[0]) * (obs[i].center[0] - x[0]) - (obs[i].center[1] - x[1]) * (obs[i].center[1] - x[1]);
             g(3 * i + 1) = rad_squared - right_side_x * right_side_x - right_side_y * right_side_y;
             g(3 * i + 2) = rad_squared - left_side_x * left_side_x - left_side_y * left_side_y;
             
             // entries of dd
             dg(3 * i, 0) = -2 * (x[0] - obs[i].center[0]);
             dg(3 * i, 1) = -2 * (x[1] - obs[i].center[1]);
             
             dg(3 * i + 1, 0) = -2 * right_side_x;
             dg(3 * i + 1, 1) = -2 * right_side_y;
             dg(3 * i + 1, 2) = -2 * right_side_x * (-quad_L * sin(x[3])) - 2 * right_side_y * (quad_L * cos(x[3]));
             
             dg(3 * i + 2, 0) = -2 * left_side_x;
             dg(3 * i + 2, 1) = -2 * left_side_y;
             dg(3 * i + 2, 2) = -2 * left_side_x * (-quad_L * sin(x[3])) - 2 * left_side_y * (quad_L * cos(x[3]));
             }
             } */
            
            
            // PLACE A MATRIX INTO M
            void AdmmSolver::placeAinM(std::vector<Triplet<double> >* tripletsMptr, Eigen::Ref<Eigen::MatrixXd> Aii, int ii) {
                // place values of A in M
                for (int row = 0; row < num_states; row++) {
                    for (int col = 0; col < num_states; col++) {
                        if (row == col){
                            tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, num_states * ii + col, -0.5 * dt * Aii(row, col) - 1));
                            tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, num_states * (ii+1) + col, -0.5 * dt * Aii(row, col) + 1));
                        } else {
                            tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, num_states * ii + col, -0.5 * dt * Aii(row, col)));
                            tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, num_states * (ii+1) + col, -0.5 * dt * Aii(row, col)));
                        }
                    }
                }
            }
            
            
            // PLACE B MATRIX INTO M
            void AdmmSolver::placeBinM(std::vector<Triplet<double> >* tripletsMptr, Eigen::Ref<Eigen::MatrixXd> Bii, int ii) {
                // should copy entries of Bii into right place in M
                for (int row = 0; row < num_states; row++) {
                    for (int col = 0; col < num_inputs; col++) {
                        tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, N * num_states + num_inputs * ii + col, -0.5 * dt * Bii(row, col)));
                        tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, N * num_states + num_inputs * (ii+1) + col, -0.5 * dt * Bii(row, col)));
                    }
                }
            }
            
            
            // CONSTRUCT C VECTOR
            void AdmmSolver::makeCVector(Eigen::Ref<Eigen::VectorXd> c, int ii, Eigen::Ref<Eigen::VectorXd> mid_state, Eigen::Ref<Eigen::VectorXd> mid_input, Eigen::Ref<Eigen::VectorXd> fii, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii, Eigen::Ref<Eigen::VectorXd> y) {
                
                Eigen::VectorXd curr_x = y.segment(ii * num_states, num_states);
                Eigen::VectorXd curr_u = y.segment(N * num_states + ii * num_inputs, num_inputs);
                Eigen::VectorXd next_x = y.segment((ii+1) * num_states, num_states);
                Eigen::VectorXd next_u = y.segment(N * num_states + (ii+1) * num_inputs, num_inputs);
                
                c.segment((ii+1) * num_states, num_states) = dt * (fii - 0.5 * Aii * (curr_x + next_x) - 0.5 * Bii * (curr_u + next_u));
            }
            
            
            // PLACE IN G
            /*
             void AdmmSolver::placeinG(int ii, std::vector<Triplet<double> >* tripletsGptr, Eigen::Ref<Eigen::MatrixXd> point_dg_x, Eigen::Ref<Eigen::MatrixXd> point_dg_u) {
             
             // place values of point dg's into G
             for (int row = 0; row < num_constraints; row++) {
             for (int col = 0; col < num_states; col++) {
             tripletsGptr->push_back(Triplet<double>(ii * num_constraints + row, ii * num_states + col, point_dg_x(row, col)));
             }
             for (int col = 0; col < num_inputs; col++) {
             tripletsGptr->push_back(Triplet<double>(ii * num_constraints + row, N * num_states + ii * num_inputs + col, point_dg_u(row, col)));
             }
             }
             } */
            
            void AdmmSolver::placeinG(std::vector<Triplet<double> >* tripletsGptr, Eigen::Ref<Eigen::MatrixXd> single_dg_x, Eigen::Ref<Eigen::MatrixXd> single_dg_u, int ii, int running_constraint_counter, int cf_length) {
                
                for (int row = 0; row < cf_length; row++) {
                    for (int col = 0; col < num_states; col++) {
                        if (single_dg_x(row, col) != 0) {
                            tripletsGptr->push_back(Triplet<double>(ii * num_constraints + running_constraint_counter + row,
                                                                    ii * num_states + col,
                                                                    single_dg_x(row, col)));
                        }
                    }
                }
                for (int row = 0; row < cf_length; row++) {
                    for (int col = 0; col < num_inputs; col++) {
                        if (single_dg_u(row, col) != 0) {
                            tripletsGptr->push_back(Triplet<double>(ii * num_constraints + running_constraint_counter + row,
                                                                    N * num_states + ii * num_inputs + col,
                                                                    single_dg_u(row, col)));
                        }
                    }
                }
            }
            
            int main(int argc, char* argv[]) {
                //gflags::ParseCommandLineFlags(&argc, &argv, true);
                return 0;
            }
            } // namespace admm_solver
        } // namespace trajectory_optimization
    } // namespace systems
} // namespace drake

