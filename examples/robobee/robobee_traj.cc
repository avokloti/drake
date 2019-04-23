#include <memory>
#include <iostream>
#include <fstream>

#include <gflags/gflags.h>

#include "drake/systems/framework/system.h"
#include "drake/systems/framework/context.h"
#include "drake/examples/robobee/robobee_plant.h"
#include "drake/math/wrap_to.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/systems/trajectory_optimization/admm_solver_weighted_v2.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"

using namespace std;

namespace drake {
    namespace examples {
        namespace robobee {
            namespace {
                typedef trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;
                
                // make robobee plant
                RobobeePlant<double>* plant = new RobobeePlant<double>();
                auto context_ptr = plant->CreateDefaultContext();
                    
                // number of states and inputs
                int num_states = context_ptr->get_num_total_states();
                int num_inputs = plant->get_input_port().size();
                
                // global obstacle parameters
                int N = 40;
                double T = 5.0;
                double dt = T/N;
                double pi = 3.14159;
                
                // set rho parameters
                double rho1 = 0.001;
                double rho2 = 10000;
                double rho3 = 10000;
                
                // initial and final states
                Eigen::VectorXd x0(num_states);
                Eigen::VectorXd xf(num_states);
                
                // matrices for running and final costs
                Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(num_states, num_states);
                Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(num_states, num_states);
                Eigen::MatrixXd R = Eigen::MatrixXd::Identity(num_inputs, num_inputs);
                
                // upper and lower bounds
                Eigen::VectorXd state_upper_bound(num_states);
                Eigen::VectorXd state_lower_bound(num_states);
                Eigen::VectorXd input_upper_bound(num_inputs);
                Eigen::VectorXd input_lower_bound(num_inputs);
                
                // prepare output file writer and control input for dynamics integration!
                ofstream output_file;
                std::string output_folder = "/Users/ira/Documents/drake/examples/robobee/output/basic/";
                
                //=============================================================================//
                
                /* WRITE STATE TO FILE */
                int writeStateToFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj) {
                    // filename
                    std::string traj_filename = output_folder + filename + "_x_" + std::to_string(trial) + ".txt";
                    
                    // open output file
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << std::endl;
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
                        output_file << std::endl;
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
                        std::cerr << "Problem opening file at " << traj_filename << std::endl;
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
                        output_file << std::endl;
                    }
                    output_file.close();
                    
                    return 0;
                }
                
                /* WRITE HEADER FILE */
                int writeHeaderFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj_x, Eigen::Ref<Eigen::MatrixXd> traj_u, double time, double tolerance, std::string solve_result) {
                    
                    // open header file
                    std::string header_filename = output_folder + filename + "_header_" + std::to_string(trial) + ".txt";
                    output_file.open(header_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << header_filename << std::endl;
                        return -1;
                    }
                    
                    // write output
                    output_file << "Next lines: N, T, x0, xf, time, feasibility 2-norm, feasibility inf-norm, constraints 2-norm, constraints inf-norm, objective value, solve result" << std::endl;
                    output_file << N << std::endl;
                    output_file << T << std::endl;
                    for (int ii = 0; ii < num_states; ii++) {
                        output_file << x0[ii] << '\t';
                    }
                    output_file << std::endl;
                    for (int ii = 0; ii < num_states; ii++) {
                        output_file << xf[ii] << '\t';
                    }
                    output_file << std::endl;
                    output_file << time << std::endl;
                    
                    // calculate integration error
                    Eigen::MatrixXd midpoint_error(num_states, N-1);
                    for (int i = 0; i < N-1; i++) {
                        Eigen::VectorXd state_value = (traj_x.col(i) + traj_x.col(i+1))/2;
                        Eigen::VectorXd input_value = (traj_u.col(i) + traj_u.col(i+1))/2;
                        
                        // calculate dynamics at midpoint
                        context_ptr->get_mutable_continuous_state().SetFromVector(state_value);
                        auto input_port_value = &context_ptr->FixInputPort(0, plant->AllocateInputVector(plant->get_input_port()));
                        input_port_value->systems::FixedInputPortValue::GetMutableVectorData<double>()->SetFromVector(input_value);
                        
                        Eigen::MatrixXd midpoint_derivative;
                        std::unique_ptr<systems::ContinuousState<double> > continuous_state(plant->AllocateTimeDerivatives());
                        plant->CalcTimeDerivatives(*context_ptr, continuous_state.get());
                        midpoint_derivative = continuous_state->CopyToVector();
                        
                        midpoint_error.col(i) = traj_x.col(i+1) - (traj_x.col(i) + dt * midpoint_derivative);
                    }
                    
                    // reshape/map matrix into vector
                    Eigen::Map<Eigen::VectorXd> error_vector(midpoint_error.data(), midpoint_error.size());
                    
                    // append to the end of the state files from before
                    output_file << error_vector.lpNorm<2>() << std::endl;
                    output_file << error_vector.lpNorm<Eigen::Infinity>() << std::endl;
                    
                    // calculate obstacle avoidance error, combined with state and input bound error
                    //Eigen::VectorXd obstacle_error = Eigen::VectorXd::Zero(N * num_obstacles);
                    Eigen::VectorXd bounds_error = Eigen::VectorXd::Zero(2 * N * (num_states + num_inputs));
                    
                    for (int i = 0; i < N; i++) {
                        /*
                        // error from obstacles
                        for (int j = 0; j < num_obstacles; j++) {
                            obstacle_error[num_obstacles * i + j] = 1 - (obstacle_center_x[j] - traj_x(0, i)) * (obstacle_center_x[j] - traj_x(0, i))/(obstacle_radii_x[j] * obstacle_radii_x[j]) - (obstacle_center_y[j] - traj_x(1, i)) * (obstacle_center_y[j] - traj_x(1, i))/(obstacle_radii_y[j] * obstacle_radii_y[j]);
                            obstacle_error[num_obstacles * i + j] = std::max(obstacle_error[num_obstacles * i + j], 0.0);
                            if (obstacle_error[num_obstacles * i + j] > 0) {
                                std::cout << "Collision of point " << i << " with obstacle " << j << std::endl;
                            }
                        } */
                        // error from state bounds
                        for (int j = 0; j < num_states; j++) {
                            int start_index = 2 * (i * num_states + j);
                            bounds_error[start_index] = std::max(traj_x(j, i) - state_upper_bound[j], 0.0);
                            bounds_error[start_index + 1] = std::max(state_lower_bound[j] - traj_x(j, i), 0.0);
                            if (bounds_error[start_index] > tolerance) {
                                std::cout << "Point " << i << " at state " << j << " violates state upper bound, by: " << bounds_error[start_index] << std::endl;
                            }
                            if (bounds_error[start_index+1] > tolerance) {
                                std::cout << "Point " << i << " at state " << j << " violates state lower bound, by: " << bounds_error[start_index+1] << std::endl;
                            }
                        }
                        for (int j = 0; j < num_inputs; j++) {
                            int start_index = 2 * N * num_states + 2 * (i * num_inputs + j);
                            bounds_error[start_index] = std::max(traj_u(j, i) - input_upper_bound[j], 0.0);
                            bounds_error[start_index + 1] = std::max(input_lower_bound[j] - traj_u(j, i), 0.0);
                            
                            if (bounds_error[start_index] > tolerance) {
                                std::cout << "Point " << i << " at input " << j << " violates input upper bound, by: " << bounds_error[start_index] << std::endl;
                            }
                            if (bounds_error[start_index+1] > tolerance) {
                                std::cout << "Point " << i << " at input " << j << " violates input lower bound, by: " << bounds_error[start_index+1] << std::endl;
                            }
                        }
                    }
                    
                    // calculate norms and write to file
                    //Eigen::VectorXd constraint_error(N * (num_obstacles + 2 * num_states + 2 * num_inputs)); constraint_error << obstacle_error, bounds_error;
                    
                    // append to the end of the output file
                    output_file << bounds_error.lpNorm<2>() << std::endl;
                    output_file << bounds_error.lpNorm<Eigen::Infinity>() << std::endl;
                    
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
                    output_file << objective << std::endl;
                    
                    // print solve result string to header
                    output_file << solve_result << std::endl;
                    
                    output_file.close();
                    return 0;
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
                
                solvers::SolutionResult solveOPT(solvers::MathematicalProgramSolverInterface* solver, std::string solver_name, double tolerance, int trial) {
                    
                    std::cout << "\n=============== Solving problem " << trial << " with " << solver_name << "!\n" << std::endl;
                    
                    systems::trajectory_optimization::MidpointTranscription traj_opt(plant, *context_ptr, N, T/N, T/N);
                    //systems::trajectory_optimization::DirectCollocation traj_opt(plant, *context_ptr, N, T/N, T/N);
                    
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
                    //traj_opt.AddRunningCost(u.dot(R * u) + (x - xf).dot(Q * (x - xf)));
                    //traj_opt.AddFinalCost((x - xf).dot(Qf * (x - xf)));
                    
                    // initialize trajectory
                    const double timespan_init = T;
                    auto traj_init_x = PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, x0});
                    traj_opt.SetInitialTrajectory(PiecewisePolynomialType(), traj_init_x);
                    
                    // set solver options
                    if (solver_name == "ipopt") {
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "tol", 1e-1);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_tol", 1e-3);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "constr_viol_tol", tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_constr_viol_tol", tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 1);
                        const std::string print_file = output_folder + "ipopt_output_" + std::to_string(trial) + ".txt";
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "file_print_level", 4);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "output_file", print_file);
                    } else if (solver_name == "snopt") {
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Scale option", 0);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major feasibility tolerance", tolerance * 0.01);
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
                    
                    // write output to files
                    writeStateToFile(solver_name, trial, xtraj);
                    writeInputToFile(solver_name, trial, utraj);
                    writeHeaderFile(solver_name, trial, xtraj, utraj, elapsed_time.count(), tolerance, solutionResultToString(result));
                    return result;
                }
                
                /* SOLVE ADMM WITH OBSTACLES */
                Eigen::MatrixXd solveADMM(systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver, std::string solver_name, double tolerance, int trial) {
                    
                    std::cout << "\n=============== Solving problem " << trial << " with " << solver_name << ": rho0 = " << rho1 << ", rho1 = " << rho2 << ", rho3 = " << rho3 << "!\n" << std::endl;
                    
                    // initialize to a line between x0 and xf
                    Eigen::VectorXd y = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                    
                    // set parameters
                    solver->setKnotPoints(N);
                    solver->setTotalTime(T);
                    solver->setFeasibilityTolerance(tolerance);
                    solver->setConstraintTolerance(tolerance);
                    solver->setObjectiveTolerance(0.1);
                    solver->setStartAndEndState(x0, xf);
                    solver->setMaxIterations(1000);
                    
                    // set RHOS
                    solver->setRho1(rho1);
                    solver->setRho2(rho2);
                    solver->setRho3(rho3);
                    
                    // add obstacle constraints
                    //solver->addInequalityConstraintToAllKnotPoints(obstacleConstraints, num_obstacles, "obstacle constraints");
                    
                    // construct cost vectors
                    Eigen::VectorXd temp_q = -Q.transpose() * xf - Q * xf;
                    Eigen::VectorXd temp_qf = -Qf.transpose() * xf - Qf * xf;
                    Eigen::VectorXd temp_r = Eigen::VectorXd::Zero((num_states + num_inputs) * N);
                    
                    // add costs
                    solver->addQuadraticRunningCostOnState(Q, temp_q);
                    solver->addQuadraticFinalCostOnState(Qf, temp_qf);
                    Eigen::MatrixXd admmR = 0.001 * R;
                    solver->addQuadraticRunningCostOnInput(admmR, temp_r);
                    
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
                    writeHeaderFile(solver_name, trial, xtraj, utraj, elapsed_time.count(), tolerance, solve_result);
                    return xtraj;
                }
                
                int do_main(int argc, char* argv[]) {
                    
                    std::cout << plant->get_num_input_ports() << std::endl;
                    
                    // x0 and xf
                    x0 << 0.2, 0.2, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                    xf << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                    
                    // state upper and lower bounds
                    state_upper_bound = Eigen::VectorXd::Ones(num_states) * 20;
                    state_lower_bound = Eigen::VectorXd::Ones(num_states) * -20;
                    state_lower_bound(0) = 0; state_lower_bound(1) = 0; state_lower_bound(2) = 0;
                    
                    // find upper and lower bounds on inputs by putting values through the written method
                    std::vector<double> bounds = plant->GetInputBounds(*context_ptr);
                    
                    input_lower_bound << bounds[0], bounds[2], bounds[4], bounds[6];
                    input_upper_bound << bounds[1], bounds[3], bounds[5], bounds[7];
                    
                    std::cout << "upper bound:\n" << input_upper_bound << std::endl;
                    std::cout << "lower bound:\n" << input_lower_bound << std::endl;
                    
                    input_upper_bound << 1, 1, 1, 0;
                    input_lower_bound << -1, -1, -1, 0;
                    
                    // make solvers
                    solvers::MathematicalProgramSolverInterface* solver_ipopt = new solvers::IpoptSolver();
                    solvers::MathematicalProgramSolverInterface* solver_snopt = new solvers::SnoptSolver();
                    systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver_admm = new systems::trajectory_optimization::admm_solver::AdmmSolverWeightedV2(plant);
                    
                    // solve! (printing to file occurs in here)
                    double tolerance = 1e-6;
                    solveADMM(solver_admm, "admm", tolerance, 0);
                    solveOPT(solver_ipopt, "ipopt", tolerance, 0);
                    solveOPT(solver_snopt, "snopt", tolerance, 0);
                    
                    // delete solver
                    delete solver_ipopt;
                    delete solver_snopt;
                    delete solver_admm;
                    
                    return 0;
                }
            }  // namespace
        }  // namespace robobee
    }  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    return drake::examples::robobee::do_main(argc, argv);
}
