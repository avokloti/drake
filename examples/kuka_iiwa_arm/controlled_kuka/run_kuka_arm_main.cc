#include <stdio.h>
#include <memory>

#include <gflags/gflags.h>
#include <assert.h>

#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
//#include "drake/examples/kuka_iiwa_arm/controlled_kuka/controlled_kuka_trajectory.h"
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
#include "drake/multibody/rigid_body_tree.h"

// from rigidbody file:
#include "drake/manipulation/util/sim_diagram_builder.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsers/urdf_parser.h"

// mine:
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"
#include "drake/systems/trajectory_optimization/admm_solver_weighted_v2.h"


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

#define SHAPED_COST 0
#define OBS 1

namespace drake {
    namespace examples {
        namespace kuka_iiwa_arm {
            namespace {
                typedef trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;
                
                const char kUrdfPath[] =
                "drake/manipulation/models/iiwa_description/urdf/"
                "iiwa14_polytope_collision.urdf";
                
                int num_states = 14;
                int num_inputs = 7;
                int num_obstacles = 1;
                
                // define time and number of points
                int N = 40;
                double T = 6.0;
                double dt = T/N;
                
                // tolerances
                double feas_tolerance = 1e-4;
                double opt_tolerance = 0.1;
                
                // initial and final states
                Eigen::VectorXd x0(num_states);
                Eigen::VectorXd xf(num_states);
                
                // matrices for running and final costs
                Eigen::MatrixXd Q;
                Eigen::MatrixXd Qf;
                Eigen::MatrixXd R;
                
                // upper and lower bounds
                Eigen::VectorXd state_lower_bound(num_states);
                Eigen::VectorXd state_upper_bound(num_states);
                Eigen::VectorXd input_lower_bound(num_inputs);
                Eigen::VectorXd input_upper_bound(num_inputs);
                
                // obstacle vectors
                Eigen::VectorXd obstacle_center_x(num_obstacles);
                Eigen::VectorXd obstacle_center_y(num_obstacles);
                Eigen::VectorXd obstacle_radii_x(num_obstacles);
                Eigen::VectorXd obstacle_radii_y(num_obstacles);
                
                // prepare trajectories
                trajectories::PiecewisePolynomial<double> xtraj_ipopt;
                trajectories::PiecewisePolynomial<double> utraj_ipopt;
                trajectories::PiecewisePolynomial<double> xtraj_snopt;
                trajectories::PiecewisePolynomial<double> utraj_snopt;
                
                // for writing files
                std::ofstream output_file;
                std::string output_folder;
                
                drake::lcm::DrakeLcm lcm;
                SimDiagramBuilder<double> builder;
                
                std::unique_ptr<RigidBodyTree<double>> tree;
                drake::systems::RigidBodyPlant<double>* rbplant;
                std::unique_ptr<systems::Context<double>> context_ptr;
                std::unique_ptr<RigidBodyTree<double>> rbtree;
                
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
                
                /* OBSTACLE CONSTRAINTS
                void obstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                    
                    // prepare for kinematics calculations
                    Eigen::VectorXd q = x.segment(0, 7);
                    Eigen::VectorXd v = x.segment(7, 7);
                    KinematicsCache<double> cache = rbtree->doKinematics(q, v);
                    Eigen::Matrix<double, 3, -1> points = Eigen::Matrix<double, 3, -1>::Zero(3, 1);
                    
                    // calculate end-effector position and Jacobian
                    Eigen::VectorXd ee = rbtree->transformPoints(cache, points, 10, 0);
                    Eigen::MatrixXd ee_jacobian = rbtree->transformPointsJacobian(cache, points, 10, 0, false);
                    
                    // place correctly in constraint matrix
                    for (int i = 0; i < num_obstacles; i++) {
                        // entries of d
                        g(i) = 1 - (obstacle_center_x[i] - ee[0]) * (obstacle_center_x[i] - ee[0])/(obstacle_radii_x[i] * obstacle_radii_x[i]) - (obstacle_center_y[i] - ee[1]) * (obstacle_center_y[i] - ee[1])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                        
                        // entries of dd
                        double aa = (ee[0] - obstacle_center_x[i])/(obstacle_radii_x[i] * obstacle_radii_x[i]);
                        double bb = (ee[1] - obstacle_center_y[i])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                        Eigen::VectorXd gradient = -2 * (aa * ee_jacobian.row(0) + bb * ee_jacobian.row(1));
                        dg_x.block(i, 0, 1, 7) = gradient.transpose();
                        //std::cout << "\n" << "--" << "\n" << dg_x << std::endl;
                    }
                } */
                
                /* OBSTACLE CONSTRAINTS (better scaling?) */
                void obstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                    
                    // prepare for kinematics calculations
                    Eigen::VectorXd q = x.segment(0, 7);
                    Eigen::VectorXd v = x.segment(7, 7);
                    KinematicsCache<double> cache = rbtree->doKinematics(q, v);
                    Eigen::Matrix<double, 3, -1> points = Eigen::Matrix<double, 3, -1>::Zero(3, 1);
                    
                    // calculate end-effector position and Jacobian
                    Eigen::VectorXd ee = rbtree->transformPoints(cache, points, 10, 0);
                    Eigen::MatrixXd ee_jacobian = rbtree->transformPointsJacobian(cache, points, 10, 0, false);
                    
                    // place correctly in constraint matrix
                    for (int i = 0; i < num_obstacles; i++) {
                        // entries of d
                        g(i) = (obstacle_radii_x[i] * obstacle_radii_x[i]) * (obstacle_radii_y[i] * obstacle_radii_y[i]) - (obstacle_center_x[i] - ee[0]) * (obstacle_center_x[i] - ee[0]) * (obstacle_radii_y[i] * obstacle_radii_y[i]) - (obstacle_center_y[i] - ee[1]) * (obstacle_center_y[i] - ee[1]) * (obstacle_radii_x[i] * obstacle_radii_x[i]);
                        
                        // entries of dd
                        double aa = (ee[0] - obstacle_center_x[i]) * (obstacle_radii_y[i] * obstacle_radii_y[i]);
                        double bb = (ee[1] - obstacle_center_y[i]) * (obstacle_radii_x[i] * obstacle_radii_x[i]);
                        Eigen::VectorXd gradient = -2 * (aa * ee_jacobian.row(0) + bb * ee_jacobian.row(1));
                        dg_x.block(i, 0, 1, 7) = gradient.transpose();
                        //std::cout << "\n" << "--" << "\n" << dg_x << std::endl;
                    }
                }
                
                Eigen::VectorXd obstacleConstraintsHelper(Eigen::Ref<Eigen::VectorXd> x) {
                    Eigen::VectorXd g(num_obstacles);
                    Eigen::VectorXd temp_u = Eigen::VectorXd::Zero(num_inputs);
                    Eigen::MatrixXd temp_dg_x(num_obstacles, num_states);
                    Eigen::MatrixXd temp_dg_u(num_obstacles, num_inputs);
                    obstacleConstraints(0, x, temp_u, g, temp_dg_x, temp_dg_u);
                    return g;
                }
                
                /* WRITE HEADER FILE */
                int writeHeaderFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj_x, Eigen::Ref<Eigen::MatrixXd> traj_u, double total_iterations, double time, double rho1, double rho2, double rho3, std::string solve_result) {
                    
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
                        auto input_port_value = &context_ptr->FixInputPort(0, rbplant->AllocateInputVector(rbplant->get_input_port(0)));
                        input_port_value->systems::FixedInputPortValue::GetMutableVectorData<double>()->SetFromVector(input_value);
                        
                        Eigen::MatrixXd midpoint_derivative;
                        std::unique_ptr<systems::ContinuousState<double> > continuous_state(rbplant->AllocateTimeDerivatives());
                        rbplant->CalcTimeDerivatives(*context_ptr, continuous_state.get());
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
                        // calculate obstacle error
                        obstacle_error.segment(i * num_obstacles, num_obstacles) = obstacleConstraintsHelper(traj_x.col(i)).cwiseMax(Eigen::VectorXd::Zero(num_obstacles));
                        
                        // error from state bounds
                        for (int j = 0; j < num_states; j++) {
                            int start_index = 2 * (i * num_states + j);
                            bounds_error[start_index] = std::max(traj_x(j, i) - state_upper_bound[j], 0.0);
                            bounds_error[start_index + 1] = std::max(state_lower_bound[j] - traj_x(j, i), 0.0);
                            if (bounds_error[start_index] > feas_tolerance) {
                                std::cout << "Point " << i << " at state " << j << " violates state upper bound by " << bounds_error[start_index] << std::endl;
                            }
                            if (bounds_error[start_index+1] > feas_tolerance) {
                                std::cout << "Point " << i << " at state " << j << " violates state lower bound by " << bounds_error[start_index+1] << std::endl;
                            }
                        }
                        for (int j = 0; j < num_inputs; j++) {
                            int start_index = 2 * N * num_states + 2 * (i * num_inputs + j);
                            bounds_error[start_index] = std::max(traj_u(j, i) - input_upper_bound[j], 0.0);
                            bounds_error[start_index + 1] = std::max(input_lower_bound[j] - traj_u(j, i), 0.0);
                            
                            if (bounds_error[start_index] > feas_tolerance) {
                                std::cout << "Point " << i << " at input " << j << " violates input upper bound by " << bounds_error[start_index] << std::endl;
                            }
                            if (bounds_error[start_index+1] > feas_tolerance) {
                                std::cout << "Point " << i << " at input " << j << " violates input lower bound by " << bounds_error[start_index+1] << std::endl;
                            }
                        }
                    }
                    
                    Eigen::VectorXd constraint_error(N * (num_obstacles + 2 * num_states + 2 * num_inputs));
                    constraint_error << obstacle_error, bounds_error;
                    
                    //std::cout << obstacle_error << std::endl;
                    
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
                
                Eigen::MatrixXd calculateEndEffectorTraj(Eigen::Ref<Eigen::MatrixXd> traj) {
                    // check input
                    assert(traj.rows() == num_states);
                    assert(traj.cols() == N);
                    
                    // make space for new values
                    Eigen::MatrixXd ee_traj = Eigen::MatrixXd::Zero(3, N);
                    Eigen::Matrix<double, 3, -1> points = Eigen::Matrix<double, 3, -1>::Zero(3, 1);
                    
                    // for each point, compute transform and Jacobian
                    for (int i = 0; i < N; i++) {
                        Eigen::VectorXd q = traj.block(0, i, 7, 1);
                        Eigen::VectorXd v = traj.block(7, i, 7, 1);
                        KinematicsCache<double> cache = rbtree->doKinematics(q, v);
                        ee_traj.col(i) = rbtree->transformPoints(cache, points, 10, 0);
                    }
                    
                    //std::cout << "ee_traj:\n" << ee_traj << std::endl;
                    return ee_traj;
                }
                
                void printEndEffectorTraj(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj) {
                    
                    Eigen::MatrixXd ee_traj = calculateEndEffectorTraj(traj);
                    
                    // filename
                    std::string traj_filename = output_folder + filename + "_ee_" + std::to_string(trial) + ".txt";
                    
                    // open output file
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << endl;
                        return;
                    }
                    
                    // write values to output file
                    for (int i = 0; i < N; i++) {
                        Eigen::VectorXd x = ee_traj.col(i);
                        output_file << i * T/(N-1) << '\t' << x[0] << '\t' << x[1] << '\t' << x[2] << endl;
                    }
                    output_file.close();
                }
                
                
                Eigen::VectorXd solveOPT(solvers::MathematicalProgramSolverInterface* solver, std::string solver_name, int trial, Eigen::Ref<Eigen::VectorXd> initial_traj) {
                    
                    std::cout << "\n=============== Solving problem " << trial << " with " << solver_name << "!\n" << std::endl;
                    
                    systems::trajectory_optimization::MidpointTranscription traj_opt(rbplant, *rbplant->CreateDefaultContext(), N, dt, dt);
                    
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
                    
                    traj_opt.AddPrintingConstraintToAllPoints(x_stream, u_stream);
                    
                    traj_opt.AddEqualTimeIntervalsConstraints(); // this wasn't here before... does it change anything?
                    
                    // get state and input placeholders
                    auto u = traj_opt.input();
                    auto x = traj_opt.state();
                    
                    // add input limits to problem
                    traj_opt.AddConstraintToAllKnotPoints(u >= input_lower_bound);
                    traj_opt.AddConstraintToAllKnotPoints(u <= input_upper_bound);
                    
                    // add joint limits to problem
                    traj_opt.AddConstraintToAllKnotPoints(x <= state_upper_bound);
                    traj_opt.AddConstraintToAllKnotPoints(x >= state_lower_bound);
                    
                    // add initial and final constraints
                    traj_opt.AddLinearConstraint(traj_opt.initial_state() == x0);
                    traj_opt.AddLinearConstraint(traj_opt.final_state() == xf);
                    
                    // add constraints to problem
                    if (SHAPED_COST) {
                        traj_opt.AddRunningCost((x - xf).dot(Q * (x - xf)));
                        traj_opt.AddFinalCost((x - xf).dot(Qf * (x - xf)));
                    }
                    traj_opt.AddRunningCost(u.dot(R * u));
                    
                    // obstacle constraint?
                    if (OBS) {
                        traj_opt.AddTaskSpaceObstacleConstraintToAllPoints(obstacle_center_x, obstacle_center_y, obstacle_radii_x, obstacle_radii_y, *rbtree);
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
                    
                    // set solver options for ipopt
                    if (solver_name == "ipopt") {
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "tol", 1e-3);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_tol", 1e-3);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "constr_viol_tol", feas_tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_constr_viol_tol", feas_tolerance);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 1);
                        const std::string print_file = output_folder + "ipopt_output_" + std::to_string(trial) + ".txt";
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "file_print_level", 4);
                        traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "output_file", print_file);
                    } else if (solver_name == "snopt") {
                        // set solver options for snopt
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Scale option", 1);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major feasibility tolerance", feas_tolerance * 0.1); // idk what to set this to
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major optimality tolerance", 1e-3);
                        traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Iterations limit", 100000);
                        const std::string print_file = output_folder + "snopt_output_" + std::to_string(trial) + ".out";
                        cout << "Should be printing to " << print_file << endl;
                        
                        // make sure output file is blank before writing to it...
                        std::ofstream ofs;
                        ofs.open(print_file, std::ofstream::out | std::ofstream::trunc);
                        ofs.close();
                        
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
                    
                    x_stream.close();
                    u_stream.close();
                    
                    calculateEndEffectorTraj(xtraj);
                    
                    // write output to files
                    writeStateToFile(solver_name, trial, xtraj);
                    writeInputToFile(solver_name, trial, utraj);
                    writeHeaderFile(solver_name, trial, xtraj, utraj, 0, elapsed_time.count(), 0, 0, 0, solutionResultToString(result));
                    printEndEffectorTraj(solver_name, trial, xtraj);
                    
                    // reshape trajectory and return
                    Eigen::VectorXd opt_traj(N * (num_states + num_inputs));
                    Map<VectorXd> xtraj_reshape(xtraj.data(), xtraj.size());
                    Map<VectorXd> utraj_reshape(utraj.data(), utraj.size());
                    opt_traj << xtraj_reshape, utraj_reshape;
                    return opt_traj;
                }
                
                Eigen::VectorXd solveADMM(systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver, std::string solver_name, double rho1, double rho2, double rho3, int trial, Eigen::Ref<Eigen::VectorXd> initial_traj) {
                    
                    std::cout << "\n=============== Solving problem " << trial << ": rho0 = " << rho1 << ", rho1 = " << rho2 << ", rho3 = " << rho3 << "!\n" << std::endl;
                    
                    // initialize
                    Eigen::VectorXd y(initial_traj);
                    
                    // set tolerances
                    solver->setKnotPoints(N);
                    solver->setTotalTime(T);
                    solver->setFeasibilityTolerance(feas_tolerance);
                    solver->setConstraintTolerance(feas_tolerance);
                    solver->setObjectiveTolerance(opt_tolerance);
                    solver->setStartAndEndState(x0, xf);
                    solver->setMaxIterations(1000);
                    
                    // set rhos
                    solver->setRho1(rho1);
                    solver->setRho2(rho2);
                    solver->setRho3(rho3);
                    
                    // add obstacle constraints
                    if (OBS) {
                        solver->addInequalityConstraintToAllKnotPoints(obstacleConstraints, num_obstacles, "obstacle constraints");
                    }
                    
                    // construct cost vectors
                    Eigen::VectorXd temp_q = -Q.transpose() * xf - Q * xf;
                    Eigen::VectorXd temp_qf = -Qf.transpose() * xf - Qf * xf;
                    Eigen::VectorXd temp_r = Eigen::VectorXd::Zero((num_states + num_inputs) * N);
                    
                    // add costs
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
                    
                    // start time and solve
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    std::string solve_result = solver->solve(y);
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    
                    // calculate time
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    std::cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    std::cout << "Solution result:" << solve_result << "\n";
                    
                    // write output to files
                    Eigen::MatrixXd xtraj = solver->getSolutionStateTrajectory();
                    Eigen::MatrixXd utraj = solver->getSolutionInputTrajectory();
                    Eigen::VectorXd ytraj = solver->getSolutionVector();
                    int total_iterations = solver->getNumLatestIterations();
                    
                    writeStateToFile(solver_name, trial, xtraj);
                    writeInputToFile(solver_name, trial, utraj);
                    writeHeaderFile(solver_name, trial, xtraj, utraj, total_iterations, elapsed_time.count(), rho1, rho2, rho3, solve_result);
                    printEndEffectorTraj(solver_name, trial, xtraj);
                    
                    return ytraj;
                }
                
                void initializePlantCostsAndBounds() {
                    // create tree, plant, builder
                    tree = std::make_unique<RigidBodyTree<double>>();
                    parsers::urdf::AddModelInstanceFromUrdfFileToWorld(FindResourceOrThrow(kUrdfPath), multibody::joints::kFixed, tree.get());
                    rbplant = builder.AddPlant(std::move(tree));
                    context_ptr = rbplant->CreateDefaultContext();
                    rbtree = rbplant->get_rigid_body_tree().Clone();
                    builder.AddVisualizer(&lcm);
                    
                    // check states and inputs
                    assert(num_states == rbplant->get_num_positions() + rbplant->get_num_velocities());
                    assert(num_inputs == rbplant->get_num_actuators());
                    
                    // set Q, Qf, R
                    if (SHAPED_COST) {
                        Q = Eigen::MatrixXd::Identity(num_states, num_states) * 0.01;
                        Qf = Eigen::MatrixXd::Identity(num_states, num_states) * 0.001;
                        R = Eigen::MatrixXd::Identity(num_inputs, num_inputs) * 0.000001;
                    } else {
                        Q = Eigen::MatrixXd::Zero(num_states, num_states);
                        // adding this in for a cost on velocities?
                        Q.block(num_states/2, num_states/2, num_states/2, num_states/2) = Eigen::MatrixXd::Identity(num_states/2, num_states/2) * 0.01;
                        Qf = Eigen::MatrixXd::Zero(num_states, num_states);
                        R = Eigen::MatrixXd::Identity(num_inputs, num_inputs) * 0.001;
                    }
                    
                    // lower and upper bounds
                    state_lower_bound << -2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054, -10, -10, -10, -10, -10, -10, -10;
                    state_upper_bound << 2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054, 10, 10, 10, 10, 10, 10, 10;
                    input_lower_bound = Eigen::VectorXd::Ones(num_inputs) * -200;
                    input_upper_bound = Eigen::VectorXd::Ones(num_inputs) * 200;
                }
                
                void initializeInitialFinalStatesAndObstacles(int trial, int random_seed) {
                    
                    // random number generator
                    std::default_random_engine generator(random_seed);
                    std::uniform_real_distribution<double> unif_dist(0, 1.0);
                    
                    if (OBS) {
                        // set x0 and xf
                        //x0 << 0, -0.683, 0, 1.77, 0, 0.88, -1.57, 0, 0, 0, 0, 0, 0, 0;
                        //xf << 0, 0, 0, -pi/4.0, 0, pi/4.0, pi/2.0, 0, 0, 0, 0, 0, 0, 0;
                        
                        x0 << -1, -1, -1, -1, -1, -1, -1.5, 0, 0, 0, 0, 0, 0, 0;
                        xf << 1, 1, 1, 1, 1, 1, 1.5, 0, 0, 0, 0, 0, 0, 0;
                        
                        //x0 << 0, 0, pi/2, pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        //xf << 0, 0, -pi/2, pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        
                        // obstacles
                        // 1. no obstacles
                        
                        // 2. away from trajectory...
                        //obstacle_center_x << 4.0;
                        //obstacle_center_y << 4.0;
                        //obstacle_radii_x << 1;
                        //obstacle_radii_y << 1;
                        
                        // 4. nearby, but does not interferes with trajectory
                        //obstacle_center_x << -0.2;
                        //obstacle_center_y << 0.05;
                        //obstacle_radii_x << 0.05;
                        //obstacle_radii_y << 0.05;
                        
                        // 4. interferes with trajectory...
                        obstacle_center_x << -0.2;
                        obstacle_center_y << 0.05;
                        obstacle_radii_x << 0.15;
                        obstacle_radii_y << 0.1;
                        
                        writeObstaclesToFile(trial);
                    } else {
                        // choose random x0 and xf (velocities set to 0)
                        for (int i = 0; i < int(num_states/2); i++) {
                            x0[i] = unif_dist(generator) * (state_upper_bound[i] - state_lower_bound[i]) + state_lower_bound[i];
                            xf[i] = unif_dist(generator) * (state_upper_bound[i] - state_lower_bound[i]) + state_lower_bound[i];
                        }
                    }
                }
                
                void runRhoSweep() {
                    if (OBS) {
                        output_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/rho_sweep_obstacles/";
                    } else {
                        output_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/rho_sweep_basic/";
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
                        initializeInitialFinalStatesAndObstacles(trial, random_seed);
                        
                        for (int i = 0; i < int(rho1_list.size()); i++) {
                            for (int j = 0; j < int(rho2_list.size()); j++) {
                                int index = trial * rho1_list.size() * rho2_list.size() + i * rho2_list.size() + j;
                                
                                // tolerances
                                double rho1 = rho1_list.at(i);
                                double rho2 = rho2_list.at(j);
                                double rho3 = rho2_list.at(j); // same as rho2
                                
                                // make solvers
                                systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver_admm = new systems::trajectory_optimization::admm_solver::AdmmSolverWeightedV2(rbplant);
                                
                                // solve! (printing to file occurs in here)
                                Eigen::VectorXd admm_traj = solveADMM(solver_admm, "admm", rho1, rho2, rho3, index, zero_traj);
                                
                                // delete solver
                                delete solver_admm;
                            }
                        }
                    }
                }
                
                void runSnoptIpoptComparison() {
                    if (OBS) {
                        output_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/test_obstacles_ws/";
                    } else {
                        output_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/alg_compare_basic/";
                    }
                    
                    // initial vectors
                    Eigen::VectorXd zero_traj = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                    
                    // SET RHO PARAMETERS
                    double rho1, rho2, rho3;
                    if (OBS) {
                        rho1 = 1;
                        rho2 = 100;
                        rho3 = 1000;
                    } else {
                        rho1 = 10;
                        rho2 = 5000;
                        rho3 = 5000;
                    }
                    
                    // number of randomized trials
                    int num_trials = 10;
                    if (OBS) {
                        num_trials = 1;
                    }
                    
                    for (int trial = 0; trial < num_trials; trial++) {
                        // make random seed
                        int random_seed = trial + 1000;
                        
                        // make new randomized instance
                        initializeInitialFinalStatesAndObstacles(trial, random_seed);
                        
                        // make solvers
                        systems::trajectory_optimization::admm_solver::AdmmSolverBase* solver_admm = new systems::trajectory_optimization::admm_solver::AdmmSolverWeightedV2(rbplant);
                        
                        solvers::MathematicalProgramSolverInterface* ipopt_solver = new solvers::IpoptSolver();
                        solvers::MathematicalProgramSolverInterface* snopt_solver = new solvers::SnoptSolver();
                        
                        // solve!
                        Eigen::VectorXd admm_sol = solveADMM(solver_admm, "admm", rho1, rho2, rho3, trial, zero_traj);
                        Eigen::VectorXd ipopt_sol = solveOPT(ipopt_solver, "ipopt", trial, admm_sol);
                        Eigen::VectorXd snopt_sol = solveOPT(snopt_solver, "snopt", trial, admm_sol);
                        
                        // delete solver
                        delete solver_admm;
                        delete snopt_solver;
                        delete ipopt_solver;
                    }
                }
                
                int do_main(int argc, char* argv[]) {
                    
                    // initialize the plant and other constant parameters that can't be declared in the outer section
                    initializePlantCostsAndBounds();
                    
                    // run comparison of ADMM against SNOPT and IPOPT
                    runSnoptIpoptComparison();
                    
                    // run parameter sweep across ADMM
                    //runRhoSweep();
                    
                    return 0;
                }
            }  // namespace
        }  // namespace kuka_iiwa_arm
    }  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    return drake::examples::kuka_iiwa_arm::do_main(argc, argv);
}
