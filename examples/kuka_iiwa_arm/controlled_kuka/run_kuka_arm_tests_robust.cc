//
//  run_kuka_arm_tests.cpp
//  
//
//  Created by Irina Tolkova on 12/14/18.
//

#include <stdio.h>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/kuka_iiwa_arm/controlled_kuka/controlled_kuka_trajectory.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/multibody_tree/multibody_plant/multibody_plant.h"
#include "drake/multibody/multibody_tree/parsing/multibody_plant_sdf_parser.h"
#include "drake/multibody/multibody_tree/uniform_gravity_field_element.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"

// from rigidbody file:
#include "drake/manipulation/util/sim_diagram_builder.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsers/urdf_parser.h"

// mine:
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"
#include "drake/systems/trajectory_optimization/admm_solver_weighted_v2.h"
#include "drake/systems/trajectory_optimization/admm_solver_al.h"
#include "drake/systems/trajectory_optimization/admm_solver_al_ineq.h"

//DEFINE_double(simulation_sec, 1, "Number of seconds to simulate.");

using drake::geometry::SceneGraph;
using drake::lcm::DrakeLcm;
using drake::multibody::Body;
using drake::multibody::multibody_plant::MultibodyPlant;
using drake::multibody::MultibodyTree;
using drake::multibody::parsing::AddModelFromSdfFile;
using drake::multibody::UniformGravityFieldElement;

// from rigidbody file:
using drake::manipulation::util::SimDiagramBuilder;
using drake::trajectories::PiecewisePolynomial;

typedef drake::trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;

#define SHAPED_COST 0
#define WARM_START_LINEAR 0
#define WARM_START_IPOPT 0
#define WARM_START_ADMM 0
#define RANDOM_START 0

namespace drake {
    namespace examples {
        namespace kuka_iiwa_arm {
            namespace {
                
                //const char kSdfPath[] = "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf";
                
                const char kUrdfPath[] =
                "drake/manipulation/models/iiwa_description/urdf/"
                "iiwa14_polytope_collision.urdf";
                
                // define pi, change this later to one of the global constants
                double pi = 3.14159;
                
                // initial and final states
                //const Eigen::VectorXd x0 = (Eigen::VectorXd(14) << 0, -0.683, 0, 1.77, 0, 0.88, -1.57, 0, 0, 0, 0, 0, 0, 0).finished();
                //const Eigen::VectorXd xf = (Eigen::VectorXd(14) << 0, -0.683, 0, 1.77, 0, 0.88, -1.5, 0, 0, 0, 0, 0, 0, 0).finished();
                //const Eigen::VectorXd xf = (Eigen::VectorXd(14) << 0, 0, 0, -pi/4.0, 0, pi/4.0, pi/2.0, 0, 0, 0, 0, 0, 0, 0).finished();
                
                Eigen::VectorXd x0(14);
                Eigen::VectorXd xf(14);
                
                int num_states = 14;
                int num_inputs = 7;
                
                // define time and number of points
                int N = 60;
                double T = 4.0;
                double dt = T/N;
                
                // matrices for running and final costs
                Eigen::MatrixXd Q;
                Eigen::MatrixXd Qf;
                Eigen::MatrixXd R;
                
                // upper and lower bounds
                Eigen::VectorXd state_lower_bound(14);
                Eigen::VectorXd state_upper_bound(14);
                Eigen::VectorXd input_lower_bound(7);
                Eigen::VectorXd input_upper_bound(7);
                
                // prepare trajectories
                trajectories::PiecewisePolynomial<double> xtraj_ipopt;
                trajectories::PiecewisePolynomial<double> utraj_ipopt;
                trajectories::PiecewisePolynomial<double> xtraj_snopt;
                trajectories::PiecewisePolynomial<double> utraj_snopt;
                
                // for writing files
                std::ofstream output_file;
                std::string output_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/random_start_end_v3/";
                
                std::unique_ptr<systems::Context<double>> context_ptr;
                
                //=============================================================================//
                
                
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
                int writeHeaderFile(drake::systems::RigidBodyPlant<double>* plant, std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj_x, Eigen::Ref<Eigen::MatrixXd> traj_u, double tolerance, double time, std::string solve_result) {
                    
                    // open header file
                    std::string header_filename = output_folder + filename + "_header_" + std::to_string(trial) + ".txt";
                    output_file.open(header_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << header_filename << endl;
                        return -1;
                    }
                    
                    // write output
                    output_file << "Next lines: N, T, x0, xf, tol, time" << endl;
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
                    output_file << tolerance << endl;
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
                    Eigen::VectorXd bounds_error = Eigen::VectorXd::Zero(2 * N * (num_states + num_inputs));
                    
                    for (int i = 0; i < N; i++) {
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
                    
                    // append to the end of the output file
                    output_file << bounds_error.lpNorm<2>() << endl;
                    output_file << bounds_error.lpNorm<Infinity>() << endl;
                    
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
                
                void initializeValues() {
                    if (SHAPED_COST) {
                        Q = Eigen::MatrixXd::Identity(num_states, num_states) * 0.01;
                        Qf = Eigen::MatrixXd::Identity(num_states, num_states) * 0.001;
                        R = Eigen::MatrixXd::Identity(num_inputs, num_inputs) * 0.000001;
                    } else {
                        Q = Eigen::MatrixXd::Zero(num_states, num_states);
                        Qf = Eigen::MatrixXd::Zero(num_states, num_states);
                        R = Eigen::MatrixXd::Identity(num_inputs, num_inputs) * 0.001;
                    }
                    
                    state_lower_bound << -2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054, -10, -10, -10, -10, -10, -10, -10;
                    state_upper_bound << 2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054, 10, 10, 10, 10, 10, 10, 10;
                    input_lower_bound = Eigen::VectorXd::Ones(num_inputs) * -200;
                    input_upper_bound = Eigen::VectorXd::Ones(num_inputs) * 200;
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
                
                Eigen::VectorXd solveOPT(drake::systems::RigidBodyPlant<double>* plant, solvers::MathematicalProgramSolverInterface* solver, std::string solver_name, double tolerance, int trial, std::string problem_type, Eigen::Ref<Eigen::MatrixXd> warm_start_traj) {
                    
                    systems::trajectory_optimization::MidpointTranscription traj_opt(plant, *plant->CreateDefaultContext(), N, dt, dt);
                    
                    // get state and input placeholders
                    auto u = traj_opt.input();
                    auto x = traj_opt.state();
                    
                    std::cout << "x size? " << x.size() << std::endl;
                    std::cout << "x(1)? " << x(1) << std::endl;
                    
                    // add input limits to problem
                    traj_opt.AddConstraintToAllKnotPoints(u >= input_lower_bound);
                    traj_opt.AddConstraintToAllKnotPoints(u <= input_upper_bound);
                    
                    // add joint limits to problem
                    traj_opt.AddConstraintToAllKnotPoints(x <= state_upper_bound);
                    traj_opt.AddConstraintToAllKnotPoints(x >= state_lower_bound);
                    
                    // add constraints to problem
                    if (SHAPED_COST) {
                        traj_opt.AddRunningCost((x - xf).dot(Q * (x - xf)));
                        traj_opt.AddFinalCost((x - xf).dot(Qf * (x - xf)));
                    }
                    traj_opt.AddRunningCost(u.dot(R * u));
                    traj_opt.AddLinearConstraint(traj_opt.initial_state() == x0);
                    traj_opt.AddLinearConstraint(traj_opt.final_state() == xf);
                    
                    // initialize trajectory
                    /*
                    auto traj_init_x = PiecewisePolynomialType::FirstOrderHold({0, T}, {Eigen::VectorXd::Zero(num_states), Eigen::VectorXd::Zero(num_states)});
                    if (WARM_START_LINEAR) {
                        traj_init_x = PiecewisePolynomialType::FirstOrderHold({0, T}, {x0, xf});
                    }
                    if (WARM_START_ADMM) {
                        traj_init_x = PiecewisePolynomial<double>::Cubic(Eigen::VectorXd::LinSpaced(N, 0, T), warm_start_traj);
                    } */
                    auto traj_init_x = PiecewisePolynomial<double>::Cubic(Eigen::VectorXd::LinSpaced(N, 0, T), warm_start_traj);
                    traj_opt.SetInitialTrajectory(PiecewisePolynomialType(), traj_init_x);
                    
                    // set solver options for ipopt
                    if (solver_name == "ipopt") {
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "tol", 1e-3);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_tol", 1e-3);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "constr_viol_tol", tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_constr_viol_tol", tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 1);
                        const std::string print_file = output_folder + "ipopt_output_" + std::to_string(trial) + ".txt";
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "file_print_level", 4);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "output_file", print_file);
                    } else if (solver_name == "snopt") {
                        // set solver options for snopt
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Scale option", 0);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major feasibility tolerance", tolerance * 0.01);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major optimality tolerance", 1e-3);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Iterations limit", 100000);
                        const std::string print_file = output_folder + "snopt_output_" + std::to_string(trial) + ".out";
                        cout << "Should be printing to " << print_file << endl;
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Print file", print_file);
                    }
                    
                    // solve and time solution
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
                    writeHeaderFile(plant, solver_name, trial, xtraj, utraj, tolerance, elapsed_time.count(), solutionResultToString(result));
                    
                    Eigen::VectorXd opt_traj(N * (num_states + num_inputs));
                    Map<VectorXd> xtraj_reshape(xtraj.data(), xtraj.size());
                    Map<VectorXd> utraj_reshape(utraj.data(), utraj.size());
                    opt_traj << xtraj_reshape, utraj_reshape;
                    return opt_traj;
                }
                
                Eigen::MatrixXd solveADMM(drake::systems::RigidBodyPlant<double>* plant, systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver, std::string solver_name, double tolerance, int trial, std::string problem_type, Eigen::Ref<Eigen::VectorXd> warm_start_traj) {
                    
                    std::cout << "\n=============== Solving problem " << problem_type << " with " << solver_name << "!\n" << std::endl;
                    
                    /*
                    if (WARM_START_LINEAR) {
                        for (int i = 0; i < N; i++) {
                            for (int ii = 0; ii < num_states; ii++) {
                                y[i * num_states + ii] = (1 - double(i)/(N-1)) * x0[ii] + double(i)/(N-1) * xf[ii];
                            }
                        }
                    }
                    
                    if (WARM_START_IPOPT) {
                        for (int i = 0; i < N * (num_states + num_inputs); i++) {
                            y[i] = warm_start_traj[i];
                        }
                    }
                    
                    if (RANDOM_START) {
                        std::default_random_engine re;
                        std::uniform_real_distribution<double> u_dist(-1, 1);
                        for (int i = 0; i < num_states; i++) {
                            std::uniform_real_distribution<double> x_dist(state_lower_bound[i], state_upper_bound[i]);
                            for (int j = 0; j < N; j++) {
                                y[j * num_states + i] = x_dist(re);
                            }
                        }
                        for (int i = N * num_states; i < N * num_states + N * num_inputs; i++) {
                            y[i] = u_dist(re);
                        }
                    } */
                    
                    // set tolerances
                    solver->setFeasibilityTolerance(tolerance);
                    solver->setKnotPoints(N);
                    solver->setStartAndEndState(x0, xf);
                    solver->setTotalTime(T);
                    
                    if (solver_name == "admm") {
                        solver->setRho1(1);
                        solver->setRho2(5000);
                        solver->setRho3(2000);
                    } else if (solver_name == "ali") {
                        solver->setRho1(100);
                        solver->setRho2(10);
                        solver->setRho3(10);
                    }
                    
                    Eigen::VectorXd temp_q = -Q.transpose() * xf - Q * xf;
                    Eigen::VectorXd temp_qf = -Qf.transpose() * xf - Qf * xf;
                    Eigen::VectorXd temp_r = Eigen::VectorXd::Zero((num_states + num_inputs) * N);
                    
                    if (SHAPED_COST) {
                        solver->addQuadraticRunningCostOnState(Q, temp_q);
                        solver->addQuadraticFinalCostOnState(Qf, temp_qf);
                    }
                    solver->addQuadraticRunningCostOnInput(R, temp_r);
                    
                    // state and input bounds
                    solver->setStateUpperBound(state_upper_bound);
                    solver->setStateLowerBound(state_lower_bound);
                    solver->setInputLowerBound(input_lower_bound);
                    solver->setInputUpperBound(input_upper_bound);
                    
                    // output file
                    solver->setOutputFile(output_folder + solver_name + "_output_" + std::to_string(trial) + ".txt");
                    solver->setTrajFile(output_folder + solver_name + "_traj_" + std::to_string(trial) + ".txt");
                    
                    // initial trajectory
                    Eigen::VectorXd y = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                    
                    // start timer
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    
                    // solve
                    std::string solve_result = solver->solve(y);
                    
                    // end timer
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    
                    std::cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    std::cout << "Solution result:" << solve_result << "\n";
                    
                    // write output to files
                    Eigen::MatrixXd xtraj_admm = solver->getSolutionStateTrajectory();
                    Eigen::MatrixXd utraj_admm = solver->getSolutionInputTrajectory();
                    
                    writeStateToFile(solver_name, trial, xtraj_admm);
                    writeInputToFile(solver_name, trial, utraj_admm);
                    writeHeaderFile(plant, solver_name, trial, xtraj_admm, utraj_admm, tolerance, elapsed_time.count(), solve_result);
                    return xtraj_admm;
                }
                
                
                int DoMain() {
                    
                    // make plant
                    //DRAKE_DEMAND(FLAGS_simulation_sec > 0);
                    
                    // create tree, plant, builder
                    auto tree = std::make_unique<RigidBodyTree<double>>();
                    CreateTreedFromFixedModelAtPose(FindResourceOrThrow(kUrdfPath), tree.get());
                    drake::lcm::DrakeLcm lcm;
                    SimDiagramBuilder<double> builder;
                    drake::systems::RigidBodyPlant<double>* plant = builder.AddPlant(std::move(tree));
                    context_ptr = plant->CreateDefaultContext();
                    builder.AddVisualizer(&lcm);
                    
                    num_states = plant->get_num_positions() + plant->get_num_velocities();
                    num_inputs = plant->get_num_actuators();
                    
                    // initialize costs and bounds
                    initializeValues();
                    
                    // define random value generators for start and end points
                    std::default_random_engine generator;
                    std::uniform_real_distribution<double> unif_dist(0, 1.0);
                    
                    for (int trial = 0; trial < 10; trial++) {
                        
                        // make solvers
                        systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver_admm = new systems::trajectory_optimization::admm_solver::AdmmSolverWeightedV2(plant);
                        
                        solvers::MathematicalProgramSolverInterface* ipopt_solver = new solvers::IpoptSolver();
                        solvers::MathematicalProgramSolverInterface* snopt_solver = new solvers::SnoptSolver();
                        
                        // fill in random x0 and xf within state bounds
                        for (int ns = 0; ns < num_states; ns++) {
                            x0[ns] = unif_dist(generator) * (state_upper_bound[ns] - state_lower_bound[ns]) + state_lower_bound[ns];
                            xf[ns] = unif_dist(generator) * (state_upper_bound[ns] - state_lower_bound[ns]) + state_lower_bound[ns];
                        }
                        
                        Eigen::VectorXd zero_traj_admm = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                        Eigen::MatrixXd zero_traj_opt = Eigen::MatrixXd::Zero(num_states, N);
                        
                        std::cout << "Iteration " << trial << " x0 and xf are: " << std::endl;
                        std::cout << x0 << std::endl;
                        std::cout << xf << std::endl;
                        
                        // solve with all methods
                        Eigen::MatrixXd admm_sol = solveADMM(plant, solver_admm, "admm", 1e-6, trial, "simple", zero_traj_admm);
                        Eigen::VectorXd ipopt_sol = solveOPT(plant, ipopt_solver, "ipopt", 1e-6, trial, "simple", zero_traj_opt);
                        Eigen::VectorXd snopt_sol = solveOPT(plant, snopt_solver, "snopt", 1e-6, trial, "simple", zero_traj_opt);
                        
                        delete solver_admm;
                        delete snopt_solver;
                        delete ipopt_solver;
                    }
                    
                    return 0;
                }
            }  // namespace
        }  // namespace kuka_iiwa_arm
    }  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    return drake::examples::kuka_iiwa_arm::DoMain();
}


// how to add obstacles...
// constraint: end effector position is outside of cylinder
// derivative of constraint:



/* ----- MULTIBODY VERSION ----- */

/*
 systems::DiagramBuilder<double> builder;
 
 SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
 scene_graph.set_name("scene_graph");
 
 // Make and add the kuka robot model.
 MultibodyPlant<double>& kuka_plant = *builder.AddSystem<MultibodyPlant>();
 AddModelFromSdfFile(FindResourceOrThrow(kSdfPath), &kuka_plant, &scene_graph);
 kuka_plant.WeldFrames(kuka_plant.world_frame(),
 kuka_plant.GetFrameByName("iiwa_link_0"));
 
 // Add gravity to the model.
 kuka_plant.AddForceElement<UniformGravityFieldElement>(
 -9.81 * Vector3<double>::UnitZ());
 
 // Now the model is complete.
 kuka_plant.Finalize(&scene_graph);
 DRAKE_THROW_UNLESS(kuka_plant.num_positions() == 7);
 // Sanity check on the availability of the optional source id before using it.
 DRAKE_DEMAND(!!kuka_plant.get_source_id());
 
 // test different quantities
 //std::cout << "num frames: " << kuka_plant.num_frames() << std::endl;
 std::cout << "num bodies: " << kuka_plant.num_bodies() << std::endl;
 std::cout << "num joints: " << kuka_plant.num_joints() << std::endl;
 std::cout << "num actuators: " << kuka_plant.num_actuators() << std::endl;
 std::cout << "num positions: " << kuka_plant.num_positions() << std::endl;
 std::cout << "num multibody states: " << kuka_plant.num_multibody_states() << std::endl;
 
 // try making transcription object from this
 systems::trajectory_optimization::MidpointTranscription traj_opt(kuka_plant, kuka_plant.CreateDefaultContext(), 10, 0.5, 0.5); */
