#include <stdio.h>
#include <memory>
#include <stdexcept>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/kuka_iiwa_arm/controlled_kuka/controlled_kuka_trajectory.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/manipulation/util/sim_diagram_builder.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"
#include "drake/systems/trajectory_optimization/admm_solver_weighted_v2.h"

#define OBS 0

using drake::manipulation::util::SimDiagramBuilder;
using drake::trajectories::PiecewisePolynomial;

namespace drake {
    namespace examples {
        namespace kuka_iiwa_arm {
            namespace {
                typedef drake::trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;
                
                const char kUrdfPath[] =
                "drake/manipulation/models/iiwa_description/urdf/"
                "iiwa14_polytope_collision.urdf";
                
                // pointer (plant is declared in body, hard to declare here)
                std::unique_ptr<systems::Context<double>> context_ptr;
                
                // number of states and inputs
                int num_states = 14;
                int num_inputs = 7;
                
                // define time and number of points
                int N = 20;
                double T = 4.0;
                double dt = T/N;
                
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
                Eigen::VectorXd state_lower_bound(num_states);
                Eigen::VectorXd state_upper_bound(num_states);
                Eigen::VectorXd input_lower_bound(num_inputs);
                Eigen::VectorXd input_upper_bound(num_inputs);
                
                // for writing files
                std::ofstream output_file;
                std::string output_folder;
                
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
                int writeHeaderFile(drake::systems::RigidBodyPlant<double>* plant, std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj_x, Eigen::Ref<Eigen::MatrixXd> traj_u, int total_iterations, double time, double rho1, double rho2, double rho3, std::string solve_result) {
                    
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
                    Eigen::VectorXd bounds_error = Eigen::VectorXd::Zero(2 * N * (num_states + num_inputs));
                    
                    for (int i = 0; i < N; i++) {
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
                
                
                Eigen::VectorXd solveADMM(drake::systems::RigidBodyPlant<double>* plant, systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver, std::string solver_name, double rho1, double rho2, double rho3, int trial, Eigen::Ref<Eigen::VectorXd> initial_traj) {
                    
                    std::cout << "\n=============== Solving problem " << trial << ": rho0 = " << rho1 << ", rho1 = " << rho2 << ", rho3 = " << rho3 << "!\n" << std::endl;
                    
                    // initialize to a line between x0 and xf
                    Eigen::VectorXd y(initial_traj);
                    
                    // set tolerances
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
                    
                    Eigen::VectorXd temp_q = -Q.transpose() * xf - Q * xf;
                    Eigen::VectorXd temp_qf = -Qf.transpose() * xf - Qf * xf;
                    Eigen::VectorXd temp_r = Eigen::VectorXd::Zero((num_states + num_inputs) * N);
                    
                    solver->addQuadraticRunningCostOnState(Q, temp_q);
                    solver->addQuadraticFinalCostOnState(Qf, temp_qf);
                    solver->addQuadraticRunningCostOnInput(R, temp_r);
                    
                    // state and input bounds
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
                    writeHeaderFile(plant, solver_name, trial, xtraj, utraj, total_iterations, elapsed_time.count(), rho1, rho2, rho3, solve_result);
                    
                    // reshape xtraj and utraj into array
                    Eigen::VectorXd xtraj_vec = Map<const VectorXd>(xtraj.data(), xtraj.size());
                    Eigen::VectorXd utraj_vec = Map<const VectorXd>(utraj.data(), utraj.size());
                    Eigen::VectorXd traj(N * (num_states + num_inputs)); traj << xtraj_vec, utraj_vec;
                    
                    return traj;
                }
                
                void initializeValues(int trial) {
                    
                    // upper and lower bounds
                    state_lower_bound << -2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054, -10, -10, -10, -10, -10, -10, -10;
                    state_upper_bound << 2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054, 10, 10, 10, 10, 10, 10, 10;
                    input_lower_bound = Eigen::VectorXd::Ones(num_inputs) * -200;
                    input_upper_bound = Eigen::VectorXd::Ones(num_inputs) * 200;
                    
                    // random number generator
                    std::default_random_engine generator(trial + 1);
                    
                    // define random value generators for start and end points
                    std::uniform_real_distribution<double> unif_dist(0, 1.0);
                    
                    if (OBS) {
                        throw std::invalid_argument("Haven't developed obstacle version yet!");
                    } else {
                        // x0 and xf
                        for (int i = 0; i < int(num_states/2); i++) {
                            x0[i] = unif_dist(generator) * (state_upper_bound[i] - state_lower_bound[i]) + state_lower_bound[i];
                            xf[i] = unif_dist(generator) * (state_upper_bound[i] - state_lower_bound[i]) + state_lower_bound[i];
                        }
                        
                        // set output folder
                        output_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/rho_sweep/";
                    }
                }
                
                
                int do_main() {
                    
                    // create tree, plant, builder
                    auto tree = std::make_unique<RigidBodyTree<double>>();
                    CreateTreedFromFixedModelAtPose(FindResourceOrThrow(kUrdfPath), tree.get());
                    SimDiagramBuilder<double> builder;
                    drake::systems::RigidBodyPlant<double>* plant = builder.AddPlant(std::move(tree));
                    context_ptr = plant->CreateDefaultContext();
                    
                    // initial trajectory
                    Eigen::VectorXd zero_traj = Eigen::VectorXd::Zero(N * (num_states + num_inputs));
                    
                    // number of randomized trials
                    int num_trials = 10;
                    
                    // make lists of rho parameters to sweep through
                    std::vector<double> rho1_list {0.01, 0.05, 0.1, 0.5, 1, 10, 100};
                    std::vector<double> rho2_list {0.1, 1, 10, 100, 1000, 5000};
                    
                    // solve
                    for (int trial = 2; trial < num_trials; trial++) {
                        
                        initializeValues(trial + num_trials);
                        
                        for (int i = 0; i < int(rho1_list.size()); i++) {
                            for (int j = 0; j < int(rho2_list.size()); j++) {
                                int index = trial * rho1_list.size() * rho2_list.size() + i * rho2_list.size() + j;
                                
                                // tolerances
                                double rho1 = rho1_list.at(i);
                                double rho2 = rho2_list.at(j);
                                double rho3 = rho2_list.at(j); // same as rho2
                    
                                // make solver
                                systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver_admm = new systems::trajectory_optimization::admm_solver::AdmmSolverWeightedV2(plant);
                                
                                // solve! (printing to file occurs in here)
                                solveADMM(plant, solver_admm, "admm", rho1, rho2, rho3, index, zero_traj);
                                
                                // delete solver
                                delete solver_admm;
                            }
                        }
                    }
                    
                    return 0;
                }
            }
        }  // namespace kuka_iiwa_arm
    }  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    return drake::examples::kuka_iiwa_arm::do_main();
}
