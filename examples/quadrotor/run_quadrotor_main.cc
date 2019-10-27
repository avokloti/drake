#include <memory>
#include <iostream>
#include <fstream>

#include <gflags/gflags.h>

#include "drake/math/wrap_to.h"
#include "drake/systems/framework/system.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/examples/quadrotor/quadrotor_plant.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/trajectory_optimization/admm_solver_weighted_v2.h"
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"

#define OBS 1

namespace drake {
    namespace examples {
        namespace quadrotor {
            namespace {
                typedef trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;
                
                // system and context
                QuadrotorPlant<double>* plant = new QuadrotorPlant<double>();
                auto context_ptr = plant->CreateDefaultContext();
                
                // number of states and inputs
                int num_states = context_ptr->get_num_total_states();
                int num_inputs = plant->get_input_port(0).size();
                
                // global obstacle parameters
                int N = 20;
                double T = 5.0;
                double dt = T/N;
                
                // tolerances
                double feas_tolerance = 1e-6;
                double opt_tolerance = 0.1;
                
                // initial and final states
                Eigen::VectorXd x0(num_states);
                Eigen::VectorXd xf(num_states);
                
                // matrices for running and final costs
                Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(num_states, num_states);
                Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(num_states, num_states);
                Eigen::MatrixXd R = Eigen::MatrixXd::Identity(num_inputs, num_inputs) * 0.001;
                
                // upper and lower bounds
                Eigen::VectorXd state_upper_bound(num_states);
                Eigen::VectorXd state_lower_bound(num_states);
                Eigen::VectorXd input_upper_bound(num_inputs);
                Eigen::VectorXd input_lower_bound(num_inputs);
                
                // arrays related to obstacles
                int num_obstacles = 4;
                int num_alpha = 10;
                Eigen::VectorXd obstacle_center_x(num_obstacles);
                Eigen::VectorXd obstacle_center_y(num_obstacles);
                Eigen::VectorXd obstacle_radii_x(num_obstacles);
                Eigen::VectorXd obstacle_radii_y(num_obstacles);
                
                // prepare output file writer and control input for dynamics integration!
                ofstream output_file;
                std::string output_folder;
                
                //=============================================================================//
                
                /* WRITE OBSTACLES TO FILE */
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
                int writeHeaderFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj_x, Eigen::Ref<Eigen::MatrixXd> traj_u, int total_iterations, double time, double rho1, double rho2, double rho3, std::string solve_result) {
                    
                    // open header file
                    std::string header_filename = output_folder + filename + "_header_" + std::to_string(trial) + ".txt";
                    output_file.open(header_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << header_filename << endl;
                        return -1;
                    }
                    
                    // write output
                    output_file << "Next lines: N, T, x0, xf, rho1, rho2, rho3, iterations, time" << endl;
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
                    output_file << feas_tolerance << endl;
                    output_file << opt_tolerance << endl;
                    output_file << total_iterations << endl;
                    output_file << time << endl;
                    
                    // calculate integration error
                    Eigen::MatrixXd midpoint_error(num_states, N-1);
                    for (int i = 0; i < N-1; i++) {
                        Eigen::VectorXd state_value = (traj_x.col(i) + traj_x.col(i+1))/2;
                        Eigen::VectorXd input_value = (traj_u.col(i) + traj_u.col(i+1))/2;
                        
                        // calculate dynamics at midpoint
                        context_ptr->get_mutable_continuous_state().SetFromVector(state_value);
                        auto input_port_value = &context_ptr->FixInputPort(0, plant->AllocateInputVector(plant->get_input_port(0)));
                        input_port_value->systems::FixedInputPortValue::GetMutableVectorData<double>()->SetFromVector(input_value);
                        
                        Eigen::MatrixXd midpoint_derivative;
                        std::unique_ptr<systems::ContinuousState<double> > continuous_state(plant->AllocateTimeDerivatives());
                        plant->CalcTimeDerivatives(*context_ptr, continuous_state.get());
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
                                std::cout << "Collision of point " << i << " with obstacle " << j << " by: " << obstacle_error[num_obstacles * i + j] << std::endl;
                            }
                        }
                        // error from state bounds
                        for (int j = 0; j < num_states; j++) {
                            int start_index = 2 * (i * num_states + j);
                            bounds_error[start_index] = std::max(traj_x(j, i) - state_upper_bound[j], 0.0);
                            bounds_error[start_index + 1] = std::max(state_lower_bound[j] - traj_x(j, i), 0.0);
                            if (bounds_error[start_index] > feas_tolerance) {
                                std::cout << "Point " << i << " at state " << j << " violates state upper bound, by: " << bounds_error[start_index] << std::endl;
                            }
                            if (bounds_error[start_index+1] > feas_tolerance) {
                                std::cout << "Point " << i << " at state " << j << " violates state lower bound, by: " << bounds_error[start_index+1] << std::endl;
                            }
                        }
                        for (int j = 0; j < num_inputs; j++) {
                            int start_index = 2 * N * num_states + 2 * (i * num_inputs + j);
                            bounds_error[start_index] = std::max(traj_u(j, i) - input_upper_bound[j], 0.0);
                            bounds_error[start_index + 1] = std::max(input_lower_bound[j] - traj_u(j, i), 0.0);
                            
                            if (bounds_error[start_index] > feas_tolerance) {
                                std::cout << "Point " << i << " at input " << j << " violates input upper bound, by: " << bounds_error[start_index] << std::endl;
                            }
                            if (bounds_error[start_index+1] > feas_tolerance) {
                                std::cout << "Point " << i << " at input " << j << " violates input lower bound, by: " << bounds_error[start_index+1] << std::endl;
                            }
                        }
                    }
                    
                    // calculate norms and write to file
                    Eigen::VectorXd constraint_error(N * (num_obstacles + 2 * num_states + 2 * num_inputs));
                    constraint_error << obstacle_error, bounds_error;
                    
                    // append to the end of the output file
                    output_file << constraint_error.lpNorm<2>() << endl;
                    output_file << constraint_error.lpNorm<Infinity>() << endl;
                    
                    // write objective value to header file
                    double objective;
                    for (int i = 0; i < N; i++) {
                        objective = objective + traj_u.col(i).transpose() * R * traj_u.col(i);
                    }
                    
                    for (int i=0; i<N-1; i++) {
                        objective = objective + (traj_x.col(i) - xf).transpose() * Q * (traj_x.col(i) - xf);
                    }
                    objective = objective + (traj_x.col(N) - xf).transpose() * Qf * (traj_x.col(N) - xf);
                    objective = objective - xf.transpose() * Qf * xf - xf.transpose() * Q * xf;
                    output_file << objective << endl;
                    
                    // print solve result string to header
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
                
                /* SOLVE WITH SNOPT OR IPOPT */
                solvers::SolutionResult solveOPT(solvers::MathematicalProgramSolverInterface* solver, std::string solver_name, int trial, Eigen::Ref<Eigen::VectorXd> initial_traj) {
                    
                    std::cout << "\n=============== Solving problem " << trial << " with " << solver_name << "!\n" << std::endl;
                    
                    systems::trajectory_optimization::MidpointTranscription traj_opt(plant, *context_ptr, N, T/N, T/N);
                    
                    // open file for writing trajectories
                    ofstream x_stream;
                    x_stream.open(output_folder + solver_name + "_traj_x_" + std::to_string(trial) + ".txt");
                    if (!x_stream.is_open()) {
                        std::cerr << "Problem opening trajectory x output file (in solveOPT).";
                    }
                    
                    ofstream u_stream;
                    u_stream.open(output_folder + solver_name + "_traj_u_" + std::to_string(trial) + ".txt");
                    if (!u_stream.is_open()) {
                        std::cerr << "Problem opening trajectory x output file (in solveOPT).";
                    }
                    
                    traj_opt.AddEqualTimeIntervalsConstraints();
                    
                    auto x = traj_opt.state();
                    auto u = traj_opt.input();
                    
                    traj_opt.AddConstraintToAllKnotPoints(x <= state_upper_bound);
                    traj_opt.AddConstraintToAllKnotPoints(x >= state_lower_bound);
                    traj_opt.AddConstraintToAllKnotPoints(u >= input_lower_bound);
                    traj_opt.AddConstraintToAllKnotPoints(u <= input_upper_bound);
                    
                    traj_opt.AddLinearConstraint(traj_opt.initial_state() == x0);
                    traj_opt.AddLinearConstraint(traj_opt.final_state() == xf);
                    
                    traj_opt.AddRunningCost(u.dot(R * u));
                    traj_opt.AddRunningCost((x - xf).dot(Q * (x - xf)));
                    //traj_opt.AddFinalCost((x - xf).dot(Qf * (x - xf)));
                    
                    //traj_opt.AddPrintingConstraintToAllPoints(x_stream, u_stream);
                    if (OBS == 1) {
                        for (int i = 0; i < num_obstacles; i++) {
                            traj_opt.AddConstraintToAllKnotPoints((x(0) - obstacle_center_x(i)) * (x(0) - obstacle_center_x(i))/(obstacle_radii_x(i) * obstacle_radii_x(i)) + (x(1) - obstacle_center_y(i)) * (x(1) - obstacle_center_y(i))/(obstacle_radii_y(i) * obstacle_radii_y(i)) >= 1);
                        }
                    } else if (OBS == 2) {
                        traj_opt.AddInterpolatedObstacleConstraintToAllPoints(obstacle_center_x, obstacle_center_y, obstacle_radii_x, obstacle_radii_y, num_alpha);
                    }
                    
                    // create initial trajectories
                    Eigen::VectorXd initial_traj_x = initial_traj.segment(0, N * num_states);
                    Eigen::VectorXd initial_traj_u = initial_traj.segment(N * num_states, N * num_inputs);
                    
                    Map<MatrixXd> initial_x(initial_traj_x.data(), num_states, N);
                    Map<MatrixXd> initial_u(initial_traj_u.data(), num_inputs, N);
                    
                    auto traj_init_x = PiecewisePolynomialType::Cubic(Eigen::VectorXd::LinSpaced(N, 0, T), initial_x);
                    auto traj_init_u = PiecewisePolynomialType::Cubic(Eigen::VectorXd::LinSpaced(N, 0, T), initial_u);
                    
                    // initialize trajectory
                    traj_opt.SetInitialTrajectory(traj_init_u, traj_init_x);
                    
                    // set solver options
                    if (solver_name == "ipopt") {
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "tol", 1e-1);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_tol", 1e-3);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "constr_viol_tol", feas_tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_constr_viol_tol", feas_tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 1);
                        const std::string print_file = output_folder + "ipopt_output_" + std::to_string(trial) + ".txt";
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "file_print_level", 4);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "output_file", print_file);
                    } else if (solver_name == "snopt") {
                        //traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Scale option", 0);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major feasibility tolerance", feas_tolerance * 0.1); // this was * 0.01 before!
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major optimality tolerance", 1e-1);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Iterations limit", 200000);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major iterations limit", 10000);
                        const std::string print_file = output_folder + "snopt_output_" + std::to_string(trial) + ".out";
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Print file", print_file);
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
                    
                    x_stream.close();
                    u_stream.close();
                    
                    // write output to files
                    writeStateToFile(solver_name, trial, xtraj);
                    writeInputToFile(solver_name, trial, utraj);
                    writeHeaderFile(solver_name, trial, xtraj, utraj, 0, elapsed_time.count(), 0, 0, 0, solutionResultToString(result));
                    return result;
                }
                
                /* SOLVE WITH ADMM */
                Eigen::VectorXd solveADMM(systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver, std::string solver_name, double rho1, double rho2, double rho3, int trial, Eigen::Ref<Eigen::VectorXd> initial_traj) {
                    
                    std::cout << "\n=============== Solving problem " << trial << ": rho0 = " << rho1 << ", rho1 = " << rho2 << ", rho3 = " << rho3 << "!\n" << std::endl;
                    
                    // initialize to a line between x0 and xf
                    Eigen::VectorXd y(initial_traj);
                    
                    // set parameters
                    solver->setKnotPoints(N);
                    solver->setTotalTime(T);
                    solver->setFeasibilityTolerance(feas_tolerance);
                    solver->setConstraintTolerance(feas_tolerance);
                    solver->setObjectiveTolerance(opt_tolerance);
                    solver->setStartAndEndState(x0, xf);
                    solver->setMaxIterations(8000);
                    
                    // set RHOS
                    solver->setRho1(rho1);
                    solver->setRho2(rho2);
                    solver->setRho3(rho3);
                    
                    // add obstacle constraints
                    if (OBS == 1) {
                        solver->addInequalityConstraintToAllKnotPoints(obstacleConstraints, num_obstacles, "obstacle constraints");
                    } else if (OBS == 2) {
                        solver->addInequalityConstraintToConsecutiveKnotPoints(interpolatedObstacleConstraints, num_obstacles * num_alpha, "interpolated obstacle constraints");
                    }
                    
                    // construct cost vectors
                    Eigen::VectorXd temp_q = -Q.transpose() * xf - Q * xf;
                    Eigen::VectorXd temp_qf = -Qf.transpose() * xf - Qf * xf;
                    Eigen::VectorXd temp_r = Eigen::VectorXd::Zero((num_states + num_inputs) * N);
                    
                    // add costs
                    solver->addQuadraticRunningCostOnState(Q, temp_q);
                    solver->addQuadraticFinalCostOnState(Qf, temp_qf);
                    solver->addQuadraticRunningCostOnInput(R, temp_r);
                    
                    // set state and input bounds
                    solver->setStateUpperBound(state_upper_bound);
                    solver->setStateLowerBound(state_lower_bound);
                    solver->setInputLowerBound(input_lower_bound);
                    solver->setInputUpperBound(input_upper_bound);
                    
                    // output file
                    solver->setOutputFile(output_folder + solver_name + "_output_" + std::to_string(trial) + ".txt");
                    solver->setTrajFile(output_folder + solver_name + "_traj_" + std::to_string(trial) + ".txt");
                    
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
                        solve_result = "ExceptionThrown";
                        xtraj = Eigen::MatrixXd::Zero(num_states, N);
                        utraj = Eigen::MatrixXd::Zero(num_inputs, N);
                        total_iterations = 0;
                    }
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    cout << "Finished! Runtime = " << elapsed_time.count() << " sec." << std::endl;
                    cout << "Solve result = " << solve_result << std::endl;
                    
                    // write data to files
                    writeStateToFile(solver_name, trial, xtraj);
                    writeInputToFile(solver_name, trial, utraj);
                    writeHeaderFile(solver_name, trial, xtraj, utraj, total_iterations, elapsed_time.count(), rho1, rho2, rho3, solve_result);
                    
                    // reshape xtraj and utraj into array
                    Eigen::VectorXd xtraj_vec = Map<const VectorXd>(xtraj.data(), xtraj.size());
                    Eigen::VectorXd utraj_vec = Map<const VectorXd>(utraj.data(), utraj.size());
                    Eigen::VectorXd traj(N * (num_states + num_inputs)); traj << xtraj_vec, utraj_vec;
                    
                    return traj;
                }
                
                void initializeValues(int trial, int random_seed) {
                    
                    // state and input bounds
                    state_upper_bound << 6.1, 6.1, 4.1, 20, 0.2, 20, 20, 20, 20, 20, 20, 20;
                    state_lower_bound << -0.1, -0.1, -0.1, -20, -0.2, -20, -20, -20, -20, -20, -20, -20;
                    input_upper_bound << 10, 10, 10, 10;
                    input_lower_bound << 0, 0, 0, 0;
                    
                    // random number generator
                    std::default_random_engine generator(random_seed);
                    std::uniform_real_distribution<double> unif_dist(0, 1.0);
                    
                    if (OBS) {
                        // fix start/end locations
                        x0 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        xf << 6, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        
                        // fill in obstacle positions and sizes
                        for (int ns = 0; ns < num_obstacles; ns++) {
                            obstacle_center_x[ns] = unif_dist(generator) * 4 + 1;
                            obstacle_center_y[ns] = unif_dist(generator) * 4 + 1;
                            obstacle_radii_x[ns] = unif_dist(generator) * 0.8 + 0.2;
                            obstacle_radii_y[ns] = unif_dist(generator) * 0.8 + 0.2;
                        }
                        
                        writeObstaclesToFile(trial);
                        
                    } else {
                        // x0 and xf
                        //for (int i = 0; i < int(num_states/2); i++) {
                        for (int i = 0; i < int(3); i++) {
                            x0[i] = unif_dist(generator) * (state_upper_bound[i] - state_lower_bound[i]) + state_lower_bound[i];
                            xf[i] = unif_dist(generator) * (state_upper_bound[i] - state_lower_bound[i]) + state_lower_bound[i];
                        }
                    }
                }
                
                
                void runRhoSweep() {
                    // set output folder
                    if (OBS == 0) {
                        output_folder = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_sweep/";
                    } else if (OBS == 1) {
                        output_folder = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_sweep_obs/";
                    } else if (OBS == 2) {
                        output_folder = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_sweep_obs_interp/";
                    }
                    
                    // initial trajectory
                    Eigen::VectorXd zero_traj = Eigen::VectorXd::Zero(N * (num_states + num_inputs));
                    
                    // number of randomized trials
                    int num_trials = 10;
                    
                    // rho parameters
                    std::vector<double> rho1_list {0.001, 0.01, 0.1, 1, 10, 100};
                    std::vector<double> rho2_list {0.1, 1, 10, 100, 1000, 5000, 10000};
                    
                    // solve
                    for (int trial = 0; trial < num_trials; trial++) {
                        // make random seed
                        int random_seed = trial + num_trials + 1;
                        
                        // make new randomized instance
                        initializeValues(trial, random_seed);
                        
                        for (int i = 0; i < int(rho1_list.size()); i++) {
                            for (int j = 0; j < int(rho2_list.size()); j++) {
                                int index = trial * rho1_list.size() * rho2_list.size() + i * rho2_list.size() + j;
                                
                                // tolerances
                                double rho1 = rho1_list.at(i);
                                double rho2 = rho2_list.at(j);
                                double rho3 = rho2_list.at(j); // same as rho2
                                
                                // make solvers
                                systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver_admm = new systems::trajectory_optimization::admm_solver::AdmmSolverWeightedV2(plant);
                                
                                // solve! (printing to file occurs in here)
                                Eigen::VectorXd admm_traj =  solveADMM(solver_admm, "admm", rho1, rho2, rho3, index, zero_traj);
                                
                                // delete solver
                                delete solver_admm;
                            }
                        }
                    }
                }
                
                void runSnoptIpoptComparison() {
                    // set output folder
                    if (OBS == 0) {
                        output_folder = "/Users/ira/Documents/drake/examples/quadrotor/output/alg_compare/";
                    } else if (OBS == 1) {
                        output_folder = "/Users/ira/Documents/drake/examples/quadrotor/output/alg_compare_obs/";
                    } else if (OBS == 2) {
                        output_folder = "/Users/ira/Documents/drake/examples/quadrotor/output/alg_compare_obs_interp/";
                    }
                    
                    // initial trajectory
                    Eigen::VectorXd zero_traj = Eigen::VectorXd::Zero(N * (num_states + num_inputs));
                    
                    // number of randomized trials
                    int num_trials = 10;
                    
                    // rho parameters
                    double rho1, rho2, rho3;
                    if (OBS) {
                        rho1 = 1;
                        rho2 = 100;
                        rho3 = 100;
                    } else {
                        rho1 = 0.1;
                        rho2 = 10;
                        rho3 = 10;
                    }
                    
                    // solve
                    for (int trial = 0; trial < num_trials; trial++) {
                        // make random seed
                        int random_seed = trial + 1000;
                        
                        // make new randomized instance
                        initializeValues(trial, random_seed);
                        
                        // make solvers
                        systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver_admm = new systems::trajectory_optimization::admm_solver::AdmmSolverWeightedV2(plant);
                        solvers::MathematicalProgramSolverInterface* solver_ipopt = new solvers::IpoptSolver();
                        solvers::MathematicalProgramSolverInterface* solver_snopt = new solvers::SnoptSolver();
                        
                        // solve! (printing to file occurs in here)
                        Eigen::VectorXd admm_traj = solveADMM(solver_admm, "admm", rho1, rho2, rho3, trial, zero_traj);
                        solveOPT(solver_ipopt, "ipopt", trial, zero_traj);
                        solveOPT(solver_snopt, "snopt", trial, zero_traj);
                        
                        // delete solver
                        delete solver_admm;
                        delete solver_snopt;
                        delete solver_ipopt;
                    }
                }
                
                int do_main(int argc, char* argv[]) {
                    
                    runSnoptIpoptComparison();
                    
                    return 0;
                }
            }  // namespace
        }  // namespace quadrotor
    }  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    return drake::examples::quadrotor::do_main(argc, argv);
}
