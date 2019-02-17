#include <memory>
#include <iostream>
#include <fstream>

#include <gflags/gflags.h>

#include "drake/systems/framework/system.h"
#include "drake/systems/framework/context.h"
#include "drake/examples/quadrotor/quadrotor_plant.h"
#include "drake/math/wrap_to.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/trajectory_optimization/admm_solver_scaled.h"
#include "drake/systems/trajectory_optimization/admm_solver_al.h"
#include "drake/systems/trajectory_optimization/admm_solver_al_ineq.h"
#include "drake/systems/trajectory_optimization/admm_solver_weighted_v2.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"


namespace drake {
    namespace examples {
        namespace quadrotor {
            namespace {
                typedef trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;
                
                // prepare output file writer and control input for dynamics integration!
                ofstream output_file;
                std::string output_folder = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_sweep/";
                
                // system and context
                QuadrotorPlant<double>* quadrotor = new QuadrotorPlant<double>();
                auto quadrotor_context_ptr = quadrotor->CreateDefaultContext();
                
                // number of states and inputs
                int num_states = quadrotor_context_ptr->get_num_total_states();
                int num_inputs = quadrotor->get_input_port(0).size();
                
                // global obstacle parameters
                double T = 5.0;
                int N = 20;
                double tolerance = 1e-6;
                
                int num_alpha = 10;
                int num_obstacles = 2;
                
                Eigen::VectorXd obstacle_center_x(num_obstacles);
                Eigen::VectorXd obstacle_center_y(num_obstacles);
                Eigen::VectorXd obstacle_radii_x(num_obstacles);
                Eigen::VectorXd obstacle_radii_y(num_obstacles);
                
                Eigen::VectorXd state_upper_bound(num_states);
                Eigen::VectorXd state_lower_bound(num_states);
                Eigen::VectorXd input_upper_bound(num_inputs);
                Eigen::VectorXd input_lower_bound(num_inputs);
                
                int writeObstaclesToFile(int trial) {
                    // write obstacles to file
                    ofstream output_obs;
                    output_obs.open(output_folder + "obstacles" + std::to_string(trial) + ".txt");
                    if (!output_obs.is_open()) {
                        cerr << "Problem opening obstacle output file.\n";
                    }
                    output_obs << obstacle_center_x.transpose() << endl;
                    output_obs << obstacle_center_y.transpose() << endl;
                    output_obs << obstacle_radii_x.transpose() << endl;
                    output_obs << obstacle_radii_y.transpose() << endl;
                    output_obs.close();
                    
                    return 0;
                }
                
                /* WRITE STATE TO FILE */
                int writeStateToFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj) {
                    // filename
                    std::string traj_filename = output_folder + filename + "_x_" + std::to_string(trial) + ".txt";
                    
                    // open output file
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << endl;
                        return -1;
                    }
                    
                    // write values to output file
                    for (int i = 0; i < N; i++) {
                        Eigen::VectorXd x = traj.col(i);
                        
                        // write time
                        output_file << i * T/(N-1) << '\t';
                        
                        // write all state values
                        for (int ii = 0; ii < num_states; ii++) {
                            output_file << x[ii] << '\t';
                        }
                        output_file << endl;
                    }
                    output_file.close();
                    
                    return 0;
                }
                
                /* WRITE INPUT TO FILE */
                int writeInputToFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj) {
                    // filename
                    std::string traj_filename = output_folder + filename + "_u_" + std::to_string(trial) + ".txt";
                    
                    // open output file
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << endl;
                        return -1;
                    }
                    
                    // write values to output file
                    for (int i = 0; i < N; i++) {
                        Eigen::VectorXd u = traj.col(i);
                        
                        // write time
                        output_file << i * T/(N-1) << '\t';
                        
                        // write all state values
                        for (int ii = 0; ii < num_inputs; ii++) {
                            output_file << u[ii] << '\t';
                        }
                        output_file << endl;
                    }
                    output_file.close();
                    
                    return 0;
                }
                
                /* WRITE HEADER FILE */
                int writeHeaderFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj_x, Eigen::Ref<Eigen::MatrixXd> traj_u, Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> xf, double rho1, double rho2, double rho3, int total_iterations, double time, std::string solve_result) {
                    
                    // open header file
                    std::string header_filename = output_folder + filename + "_header_" + std::to_string(trial) + ".txt";
                    output_file.open(header_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << header_filename << endl;
                        return -1;
                    }
                    
                    // write output
                    output_file << "Next lines: N, T, x0, xf, rho1, rho2, rho3, iterations, time, 2-norm feasibility error, inf-norm feasibility error, 2-norm constraint error, inf-norm constraint error, solve_result" << endl;
                    output_file << N << endl;
                    output_file << T << endl;
                    for (int ii = 0; ii < num_states; ii++) {
                        output_file << x0[ii] << '\t';
                    }
                    output_file << endl;
                    for (int ii = 0; ii < num_states; ii++) {
                        output_file << xf[ii] << '\t';
                    }
                    output_file << endl;
                    output_file << rho1 << endl;
                    output_file << rho2 << endl;
                    output_file << rho3 << endl;
                    output_file << total_iterations << endl;
                    output_file << time << endl;
                    
                    double dt = T/N;
                    
                    // calculate integration error
                    Eigen::MatrixXd midpoint_error(num_states, N-1);
                    for (int i = 0; i < N-1; i++) {
                        Eigen::VectorXd state_value = (traj_x.col(i) + traj_x.col(i+1))/2;
                        Eigen::VectorXd input_value = (traj_u.col(i) + traj_u.col(i+1))/2;
                        
                        // calculate dynamics at midpoint
                        quadrotor_context_ptr->get_mutable_continuous_state().SetFromVector(state_value);
                        auto input_port_value = &quadrotor_context_ptr->FixInputPort(0, quadrotor->AllocateInputVector(quadrotor->get_input_port(0)));
                        input_port_value->systems::FixedInputPortValue::GetMutableVectorData<double>()->SetFromVector(input_value);
                        
                        Eigen::MatrixXd midpoint_derivative;
                        std::unique_ptr<systems::ContinuousState<double> > continuous_state(quadrotor->AllocateTimeDerivatives());
                        quadrotor->CalcTimeDerivatives(*quadrotor_context_ptr, continuous_state.get());
                        midpoint_derivative = continuous_state->CopyToVector();
                        
                        midpoint_error.col(i) = traj_x.col(i+1) - (traj_x.col(i) + dt * midpoint_derivative);
                    }
                    
                    // reshape/map matrix into vector
                    Map<VectorXd> error_vector(midpoint_error.data(), midpoint_error.size());
                    
                    // append to the end of the state files from before
                    output_file << error_vector.lpNorm<2>() << endl;
                    output_file << error_vector.lpNorm<Infinity>() << endl;
                    
                    // calculate obstacle avoidance error, combined with state and input bound error
                    Eigen::VectorXd obstacle_error = Eigen::VectorXd::Zero(N * num_obstacles);
                    Eigen::VectorXd bounds_error = Eigen::VectorXd::Zero(2 * N * (num_states + num_inputs));
                    
                    for (int i = 0; i < N; i++) {
                        // error from obstacles
                        for (int j = 0; j < num_obstacles; j++) {
                            obstacle_error[num_obstacles * i + j] = 1 - (obstacle_center_x[j] - traj_x(0, i)) * (obstacle_center_x[j] - traj_x(0, i))/(obstacle_radii_x[j] * obstacle_radii_x[j]) - (obstacle_center_y[j] - traj_x(1, i)) * (obstacle_center_y[j] - traj_x(1, i))/(obstacle_radii_y[j] * obstacle_radii_y[j]);
                            obstacle_error[num_obstacles * i + j] = std::max(obstacle_error[num_obstacles * i + j], 0.0);
                            if (obstacle_error[num_obstacles * i + j] > 0) {
                                std::cout << "Collision of point " << i << " with obstacle " << j << ": " << obstacle_error[num_obstacles * i + j] << std::endl;
                            }
                        }
                        // error from state bounds
                        for (int j = 0; j < num_states; j++) {
                            int start_index = 2 * (i * num_states + j);
                            bounds_error[start_index] = std::max(traj_x(j, i) - state_upper_bound[j], 0.0);
                            bounds_error[start_index + 1] = std::max(state_lower_bound[j] - traj_x(j, i), 0.0);
                            if (bounds_error[start_index] > 0) {
                                std::cout << "Point " << i << " at state " << j << "violates state upper bound: " << traj_x(j, i) << std::endl;
                            }
                            if (bounds_error[start_index+1] > 0) {
                                std::cout << "Point " << i << " at state " << j << "violates state lower bound: " << traj_x(j, i) << std::endl;
                            }
                        }
                        for (int j = 0; j < num_inputs; j++) {
                            int start_index = 2 * N * num_states + 2 * (i * num_inputs + j);
                            bounds_error[start_index] = std::max(traj_u(j, i) - input_upper_bound[j], 0.0);
                            bounds_error[start_index + 1] = std::max(input_lower_bound[j] - traj_u(j, i), 0.0);
                            
                            if (bounds_error[start_index] > 0) {
                                std::cout << "Point " << i << " violates input upper bound" << std::endl;
                            }
                            if (bounds_error[start_index+1] > 0) {
                                std::cout << "Point " << i << " violates input lower bound" << std::endl;
                            }
                        }
                    }
                    
                    // calculate norms and write to file
                    Eigen::VectorXd constraint_error(N * (num_obstacles + 2 * num_states + 2 * num_inputs)); constraint_error << obstacle_error, bounds_error;
                    
                    //std::cout << "Constraint error vector: " << constraint_error << std::endl;
                    
                    // append to the end of the output file
                    output_file << constraint_error.lpNorm<2>() << endl;
                    output_file << constraint_error.lpNorm<Infinity>() << endl;
                    
                    // write objective value to header file
                    double objective;
                    for (int i = 0; i < N; i++) {
                        objective = objective + traj_u.col(i).transpose() * traj_u.col(i);
                    }
                    output_file << objective << endl;
                    output_file << solve_result << endl;
                    
                    output_file.close();
                    return 0;
                }
                
                
                /* OBSTACLE CONSTRAINTS */
                void obstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                    
                    for (int i = 0; i < num_obstacles; i++) {
                        // entries of d
                        g(i) = 1 - (obstacle_center_x[i] - x[0]) * (obstacle_center_x[i] - x[0])/(obstacle_radii_x[i] * obstacle_radii_x[i]) - (obstacle_center_y[i] - x[1]) * (obstacle_center_y[i] - x[1])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                        
                        // entries of dd
                        dg_x(i, 0) = -2 * (x[0] - obstacle_center_x[i])/(obstacle_radii_x[i] * obstacle_radii_x[i]);
                        dg_x(i, 1) = -2 * (x[1] - obstacle_center_y[i])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                    }
                }
                
                
                /* INTERPOLATED OBSTACLE CONSTRAINTS */
                void interpolatedObstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x1, Eigen::Ref<Eigen::VectorXd> u1, Eigen::Ref<Eigen::VectorXd> x2, Eigen::Ref<Eigen::VectorXd> u2, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x1, Eigen::Ref<Eigen::MatrixXd> dg_u1, Eigen::Ref<Eigen::MatrixXd> dg_x2, Eigen::Ref<Eigen::MatrixXd> dg_u2) {
                    
                    //std::cout << "In interpolated obs con" << std::endl;
                    //int num_alpha = 10;
                    std::vector<double> alpha;
                    
                    for (int i = 0; i < num_alpha; i++) {
                        alpha.push_back(static_cast<double>(i)/static_cast<double>(num_alpha));
                    }
                    
                    for (int i = 0; i < num_obstacles; i++) {
                        for (int ii = 0; ii < num_alpha; ii++) {
                            // entries of d
                            int index = i * num_alpha + ii;
                            g(index) = 1 -
                            ((1 - alpha[ii]) * x1[0] + alpha[ii] * x2[0] - obstacle_center_x[i]) *
                            ((1 - alpha[ii]) * x1[0] + alpha[ii] * x2[0] - obstacle_center_x[i])/(obstacle_radii_x[i] * obstacle_radii_x[i]) -
                            ((1 - alpha[ii]) * x1[1] + alpha[ii] * x2[1] - obstacle_center_y[i]) *
                            ((1 - alpha[ii]) * x1[1] + alpha[ii] * x2[1] - obstacle_center_y[i])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                            
                            // entries of dd
                            dg_x1(index, 0) = -2 * (1 - alpha[ii]) * ((1 - alpha[ii]) * x1[0] + alpha[ii] * x2[0] - obstacle_center_x[i])/(obstacle_radii_x[i] * obstacle_radii_x[i]);
                            dg_x1(index, 1) = -2 * (1 - alpha[ii]) * ((1 - alpha[ii]) * x1[1] + alpha[ii] * x2[1] - obstacle_center_y[i])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                            dg_x2(index, 0) = -2 * alpha[ii] * ((1 - alpha[ii]) * x1[0] + alpha[ii] * x2[0] - obstacle_center_x[i])/(obstacle_radii_x[i] * obstacle_radii_x[i]);
                            dg_x2(index, 1) = -2 * alpha[ii] * ((1 - alpha[ii]) * x1[1] + alpha[ii] * x2[1] - obstacle_center_y[i])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                        }
                    }
                }
                
                std::string solutionResultToString(solvers::SolutionResult result) {
                    std::string result_str;
                    if (result == solvers::SolutionResult::kSolutionFound) {
                        result_str = "SolutionFound";
                    } else if (result == solvers::SolutionResult::kInvalidInput) {
                        result_str = "InvalidInput";
                    } else if (result == solvers::SolutionResult::kInfeasibleConstraints) {
                        result_str = "InfeasibleConstraints";
                    } else if (result == solvers::SolutionResult::kUnbounded) {
                        result_str = "Unbounded";
                    } else if (result == solvers::SolutionResult::kUnknownError) {
                        result_str = "UnknownError";
                    } else if (result == solvers::SolutionResult::kInfeasible_Or_Unbounded) {
                        result_str = "Infeasible_Or_Unbounded";
                    } else if (result == solvers::SolutionResult::kIterationLimit) {
                        result_str = "IterationLimit";
                    } else if (result == solvers::SolutionResult::kDualInfeasible) {
                        result_str = "DualInfeasible";
                    } else {
                        result_str = "UnknownSolutionResult";
                    }
                    return result_str;
                }
                
                solvers::SolutionResult solveOPT(solvers::MathematicalProgramSolverInterface* solver, std::string solver_name, Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> xf, double rho1, double rho2, double rho3, int trial_index) {
                    
                    std::cout << "\n ------------------------- Solving problem " << trial_index << " with " << solver_name << " -------------------------" << std::endl;
                    
                    systems::trajectory_optimization::MidpointTranscription traj_opt(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    
                    traj_opt.AddLinearConstraint(traj_opt.initial_state() == x0);
                    traj_opt.AddLinearConstraint(traj_opt.final_state() == xf);
                    traj_opt.AddEqualTimeIntervalsConstraints();
                    
                    auto x = traj_opt.state();
                    auto u = traj_opt.input();
                    
                    traj_opt.AddConstraintToAllKnotPoints(x <= state_upper_bound);
                    traj_opt.AddConstraintToAllKnotPoints(x >= state_lower_bound);
                    traj_opt.AddConstraintToAllKnotPoints(u >= input_lower_bound);
                    traj_opt.AddConstraintToAllKnotPoints(u <= input_upper_bound);
                    
                    const Eigen::Matrix4d R = Eigen::MatrixXd::Identity(4, 4);
                    traj_opt.AddRunningCost(u.dot(R * u));
                    
                    const double timespan_init = T;
                    
                    // initialize trajectory
                    auto traj_init_x = PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, x0});
                    traj_opt.SetInitialTrajectory(PiecewisePolynomialType(), traj_init_x);
                    
                    // set solver options for ipopt
                    if (solver_name == "ipopt") {
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "tol", 1e-3);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_tol", 1e-3);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "constr_viol_tol", tolerance * tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_constr_viol_tol", tolerance * tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 1);
                        const std::string print_file = output_folder + "ipopt_output_" + std::to_string(trial_index) + ".txt";
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "file_print_level", 4);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "output_file", print_file);
                    } else if (solver_name == "snopt") {
                        // set solver options for snopt
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Scale option", 0);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major feasibility tolerance", tolerance);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major optimality tolerance", 1e-3);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Iterations limit", 100000);
                        const std::string print_file = output_folder + "snopt_output_" + std::to_string(trial_index) + ".out";
                        cout << "Should be printing to " << print_file << endl;
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Print file", print_file);
                    }
                    
                    // add obstacle constraints!
                    for (int i = 0; i < num_obstacles; i++) {
                        traj_opt.AddConstraintToAllKnotPoints((x(0) - obstacle_center_x(i)) * (x(0) - obstacle_center_x(i))/(obstacle_radii_x(i) * obstacle_radii_x(i)) + (x(1) - obstacle_center_y(i)) * (x(1) - obstacle_center_y(i))/(obstacle_radii_y(i) * obstacle_radii_y(i)) >= 1);
                    }
                    
                    // solve!
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    solvers::SolutionResult result = solver->Solve(traj_opt);
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    std::cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    std::cout << "Solution result:" << result << "\n";
                    
                    // get output
                    Eigen::MatrixXd xtraj = traj_opt.getStateTrajectoryMatrix(num_states);
                    Eigen::MatrixXd utraj = traj_opt.getInputTrajectoryMatrix(num_inputs);
                    
                    // write output to files
                    writeStateToFile(solver_name, trial_index, xtraj);
                    writeInputToFile(solver_name, trial_index, utraj);
                    writeHeaderFile(solver_name, trial_index, xtraj, utraj, x0, xf, rho1, rho2, rho3, -1, elapsed_time.count(), solutionResultToString(result));
                    return result;
                }
                
                
                /* SOLVE ADMM WITH OBSTACLES */
                void solveADMM(systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver, std::string solver_name, Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> xf, double rho1, double rho2, double rho3, int trial_index) {
                    
                    std::cout << "\n ------------------------- Solving problem " << trial_index << " with " << solver_name << " -------------------------" << std::endl;
                    
                    std::cout << "rhos: " << rho1 << ", " << rho2 << ", " << rho3 << std::endl;
                    
                    // initialize to a line between x0 and xf
                    Eigen::VectorXd y = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                    
                    // set parameters
                    solver->setKnotPoints(N);
                    solver->setTotalTime(T);
                    solver->setFeasibilityTolerance(tolerance);
                    solver->setConstraintTolerance(tolerance);
                    solver->setObjectiveTolerance(0.1);
                    solver->setStartAndEndState(x0, xf);
                    solver->setMaxIterations(2000);
                    
                    // rhos
                    solver->setRho1(rho1);
                    solver->setRho2(rho2);
                    solver->setRho3(rho3);
                    
                    // output files
                    solver->setOutputFile(output_folder + solver_name + "_output_" + std::to_string(trial_index) + ".txt");
                    solver->setTrajFile(output_folder + solver_name + "_traj_" + std::to_string(trial_index) + ".txt");
                    
                    // add obstacle constraints
                    solver->addInequalityConstraintToAllKnotPoints(obstacleConstraints, num_obstacles, "obstacle constraints");
                    
                    // set state and input bounds
                    solver->setStateUpperBound(state_upper_bound);
                    solver->setStateLowerBound(state_lower_bound);
                    solver->setInputLowerBound(input_lower_bound);
                    solver->setInputUpperBound(input_upper_bound);
                    
                    // start timer
                    std::string solve_result;
                    Eigen::MatrixXd xtraj;
                    Eigen::MatrixXd utraj;
                    int total_iterations;
                    
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    try {
                        solve_result = solver->solve(y);
                        xtraj = solver->getSolutionStateTrajectory();
                        utraj = solver->getSolutionInputTrajectory();
                        total_iterations = solver->getNumLatestIterations();
                    } catch (std::exception e) {
                        solve_result = "DIVERGENCE";
                        xtraj = Eigen::MatrixXd::Zero(num_states, N);
                        utraj = Eigen::MatrixXd::Zero(num_inputs, N);
                        total_iterations = 0;
                    }
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    cout << "Finished! Runtime = " << elapsed_time.count() << " sec." << std::endl;
                    cout << "Solve result = " << solve_result << std::endl;
                    
                    // get xtraj and utraj
                    //Eigen::MatrixXd xtraj = solver->getSolutionStateTrajectory();
                    //Eigen::MatrixXd utraj = solver->getSolutionInputTrajectory();
                    //int total_iterations = solver->getNumLatestIterations();
                    //Eigen::MatrixXd xtraj = Eigen::MatrixXd::Zero(num_states, N);
                    //Eigen::MatrixXd utraj = Eigen::MatrixXd::Zero(num_inputs, N);
                    //int total_iterations = 0;
                    
                    // write info to files
                    writeStateToFile(solver_name, trial_index, xtraj);
                    writeInputToFile(solver_name, trial_index, utraj);
                    writeHeaderFile(solver_name, trial_index, xtraj, utraj, x0, xf, rho1, rho2, rho3, total_iterations, elapsed_time.count(), solve_result);
                }
                
                
                int do_main(int argc, char* argv[]) {
                    
                    state_upper_bound << 20, 20, 20, 20, 0.2, 20, 20, 20, 20, 20, 20, 20;
                    state_lower_bound << -20, -20, -20, -20, -0.2, -20, -20, -20, -20, -20, -20, -20;
                    input_upper_bound << 10, 10, 10, 10;
                    input_lower_bound << 0, 0, 0, 0;
                    
                    int num_trials = 5;
                    
                    std::vector<double> rho1s {1, 10, 25, 50, 75, 100, 250, 500, 750, 1000};
                    std::vector<double> rho2s {10, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
                    std::vector<double> rho3s {10, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
                    
                    // make random generators for start/end points
                    /*
                    std::default_random_engine generator;
                    std::uniform_real_distribution<double> x0_position_distribution(-6.0, -4.0);
                    std::uniform_real_distribution<double> x0_velocity_distribution(-0.2, 0.2);
                    std::uniform_real_distribution<double> xf_position_distribution(4.0, 6.0);
                    std::uniform_real_distribution<double> xf_velocity_distribution(-0.2, 0.2);
                    
                    // make random generators for obstacles
                    std::uniform_real_distribution<double> x_distribution(-3.0, 3.0);
                    std::uniform_real_distribution<double> y_distribution(-3.0, 3.0);
                    std::uniform_real_distribution<double> radii_x_distribution(0.4, 1.0);
                    std::uniform_real_distribution<double> radii_y_distribution(0.4, 1.0);
                    
                    // arrays for all randomized values
                    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(num_states);
                    Eigen::VectorXd xf = Eigen::VectorXd::Zero(num_states);
                    obstacle_center_x = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_center_y = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_radii_x = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_radii_y = Eigen::VectorXd::Zero(num_obstacles); */
                    
                    Eigen::VectorXd x0 = (Eigen::VectorXd(12) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
                    Eigen::VectorXd xf = (Eigen::VectorXd(12) << 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
                    
                    obstacle_center_x << 3, 4;
                    obstacle_center_y << 3, 4;
                    obstacle_radii_x << 0.5, 0.7;
                    obstacle_radii_y << 0.5, 0.7;
                    
                    for (int index = 562; index < 563; index++) {//int(rho1s.size() * rho2s.size() * rho3s.size()); index++) {
                        double rho3 = rho3s[index % int(rho3s.size())];
                        double rho2 = rho2s[(index/int(rho3s.size())) % int(rho2s.size())];
                        double rho1 = rho1s[(index/(int(rho3s.size()) * int(rho2s.size()))) % int(rho1s.size())];
                        
                        // randomly sample start and end position within boxes
                        /*
                        for (int ns = 0; ns < 3; ns++) {
                            x0[ns] = x0_position_distribution(generator);
                            x0[ns + 6] = x0_velocity_distribution(generator);
                            xf[ns] = xf_position_distribution(generator);
                            xf[ns + 6] = xf_velocity_distribution(generator);
                        }
                        
                        // randomly sample several obstacles (make sure to avoid start/end)
                        
                        for (int ns = 0; ns < num_obstacles; ns++) {
                            obstacle_center_x[ns] = x_distribution(generator);
                            obstacle_center_y[ns] = y_distribution(generator);
                            obstacle_radii_x[ns] = radii_x_distribution(generator);
                            obstacle_radii_y[ns] = radii_y_distribution(generator);
                        } */
                        
                        // initialize solvers
                        systems::trajectory_optimization::admm_solver::AdmmSolverBase* admm_scaled = new systems::trajectory_optimization::admm_solver::AdmmSolverScaled(quadrotor);
                        
                        systems::trajectory_optimization::admm_solver::AdmmSolverBase* admm_w = new systems::trajectory_optimization::admm_solver::AdmmSolverWeightedV2(quadrotor);
                        
                        // solve problem with all solvers
                        //solveADMM(admm_scaled, "admm_sc", x0, xf, rho1, rho2, rho3, index);
                        solveADMM(admm_w, "admm_w", x0, xf, rho1, rho2, rho3, index);
                        
                        //writeObstaclesToFile(index);
                        
                        delete admm_scaled;
                        
                        // ok, don't do scaling
                        // do paramater sweep over rho's
                        // for each combination, repeat five times and report means/sds
                        //
                    }
                    
                    return 0;
                }
            }  // namespace
        }  // namespace quadrotor
    }  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    return drake::examples::quadrotor::do_main(argc, argv);
}

