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
#include "drake/systems/trajectory_optimization/admm_solver.h"
#include "drake/systems/trajectory_optimization/accel_admm_solver.h"
#include "drake/systems/trajectory_optimization/admm_solver_weighted.h"
#include "drake/systems/trajectory_optimization/accel_weighted_admm_solver.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"

#define LINEAR_WARM_START 0
#define BUGTRAP 1

namespace drake {
    namespace examples {
        namespace quadrotor {
            namespace {
                
                typedef trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;
                
                // system and context
                QuadrotorPlant<double>* quadrotor = new QuadrotorPlant<double>();
                auto quadrotor_context_ptr = quadrotor->CreateDefaultContext();
                
                // state start and end
                const Eigen::VectorXd x0 = (Eigen::VectorXd(12) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
                const Eigen::VectorXd xf = (Eigen::VectorXd(12) << 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
                //const Eigen::VectorXd x0 = (Eigen::VectorXd(12) << -2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
                //const Eigen::VectorXd xf = (Eigen::VectorXd(12) << 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
                
                Eigen::VectorXd state_upper_bound(12);
                Eigen::VectorXd state_lower_bound(12);
                Eigen::VectorXd input_lower_bound(4);
                Eigen::VectorXd input_upper_bound(4);
                
                double T = 5.0;
                int N = 30;
                double dt = T/N;
                
                int num_states = quadrotor_context_ptr->get_num_total_states();
                int num_inputs = quadrotor->get_input_port(0).size();
                
                // prepare output file writer and control input for dynamics integration!
                ofstream output_file;
                std::string output_folder = "/Users/ira/Documents/drake/examples/quadrotor/output/forest/";
                trajectories::PiecewisePolynomial<double> dynamics_tau;
                
                // MAKE A LIST OF CYLINDRICAL OBSTACLES TO AVOID
                int num_obstacles;
                Eigen::VectorXd obstacle_center_x;
                Eigen::VectorXd obstacle_center_y;
                Eigen::VectorXd obstacle_radii_x;
                Eigen::VectorXd obstacle_radii_y;
                
                //assigns values to obstacles and state lower/upper bounds
                void initialize() {
                    if (BUGTRAP) {
                        num_obstacles = 3;
                    } else {
                        num_obstacles = 2;
                    }
                    
                    obstacle_center_x = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_center_y = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_radii_x = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_radii_y = Eigen::VectorXd::Zero(num_obstacles);
                    
                    if (BUGTRAP) {
                        obstacle_center_x << 0, 1, 1;
                        obstacle_center_y << 0, -2, 2;
                        obstacle_radii_x << 0.5, 1, 1;
                        obstacle_radii_y << 2, 0.2, 0.2;
                    } else {
                        obstacle_center_x << 0, 1.0;
                        obstacle_center_y << 1.5, 2.5;
                        obstacle_radii_x << 0.5, 0.7;
                        obstacle_radii_y << 0.5, 0.7;
                    }
                    
                    state_upper_bound << 200, 200, 200, 200, 0.2, 200, 200, 200, 200, 200, 200, 200;
                    state_lower_bound << -200, -200, -200, -200, -0.2, -200, -200, -200, -200, -200, -200, -200;
                    input_lower_bound << 0, 0, 0, 0;
                    input_upper_bound << 10, 10, 10, 10;
                    
                    // write obstacles to file
                    ofstream output_obs;
                    output_obs.open(output_folder + "obstacles_ellipse.txt");
                    if (!output_obs.is_open()) {
                        cerr << "Problem opening obstacle output file.\n";
                    }
                    output_obs << obstacle_center_x.transpose() << endl;
                    output_obs << obstacle_center_y.transpose() << endl;
                    output_obs << obstacle_radii_x.transpose() << endl;
                    output_obs << obstacle_radii_y.transpose() << endl;
                    output_obs.close();
                    
                    //state_upper_bound << 100, 100, 100, 100;
                    //state_lower_bound << -100, -100, -100, -100;
                }
                
                /* WRITE STATE TO FILE */
                int writeStateToFile(std::string filename, int trial, trajectories::PiecewisePolynomial<double> traj) {
                    // filename
                    std::string traj_filename = output_folder + filename + "_" + std::to_string(trial) + ".txt";
                    
                    // open output file
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << endl;
                        return -1;
                    }
                    
                    // write values to output file
                    for (int i = 0; i < N; i++) {
                        Eigen::VectorXd x = traj.value(i * T/(N-1));
                        
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
                int writeInputToFile(std::string filename, int trial, trajectories::PiecewisePolynomial<double> traj) {
                    // filename
                    std::string traj_filename = output_folder + filename + "_" + std::to_string(trial) + ".txt";
                    
                    // open output file
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << endl;
                        return -1;
                    }
                    
                    // write values to output file
                    for (int i = 0; i < N; i++) {
                        Eigen::VectorXd x = traj.value(i * T/(N-1));
                        
                        // write time
                        output_file << i * T/(N-1) << '\t';
                        
                        // write all state values
                        for (int ii = 0; ii < num_inputs; ii++) {
                            output_file << x[ii] << '\t';
                        }
                        output_file << endl;
                    }
                    output_file.close();
                    
                    return 0;
                }
                
                int writeOriginalTrajToFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj_x, Eigen::Ref<Eigen::MatrixXd> traj_u) {
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
                        // write time
                        output_file << i * T/(N-1) << '\t';
                        
                        // write all state values
                        for (int ii = 0; ii < num_states; ii++) {
                            output_file << traj_x(i, ii) << '\t';
                        }
                        output_file << endl;
                    }
                    output_file.close();
                    
                    // filename
                    traj_filename = output_folder + filename + "_u_" + std::to_string(trial) + ".txt";
                    
                    // open output file
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << endl;
                        return -1;
                    }
                    
                    // write values to output file
                    for (int i = 0; i < N; i++) {
                        // write time
                        output_file << i * T/(N-1) << '\t';
                        
                        // write all state values
                        for (int ii = 0; ii < num_inputs; ii++) {
                            output_file << traj_u(i, ii) << '\t';
                        }
                        output_file << endl;
                    }
                    output_file.close();
                    
                    return 0;
                }
                /*
                // WRITE HEADER FILE
                int writeHeaderFile(std::string filename, int trial, Eigen::Ref<Eigen::MatrixXd> traj_x, Eigen::Ref<Eigen::MatrixXd> traj_u, double time, double max_iter) {
                    
                    // open header file
                    std::string header_filename = output_folder + filename + "_" + std::to_string(trial) + ".txt";
                    output_file.open(header_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << header_filename << endl;
                        return -1;
                    }
                    
                    // write output
                    output_file << "Next lines: N, T, x0, xf, max_iter, time" << endl;
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
                    output_file << max_iter << endl;
                    output_file << time << endl;
                    
                    // calculate integration error
                    Eigen::MatrixXd midpoint_error(num_states, N-1);
                    for (int i = 0; i < N-1; i++) {
                        Eigen::VectorXd state_value = (traj_x.block(i, ) + traj_x.value(dt * (i+1)))/2;
                        Eigen::VectorXd input_value = (traj_u.value(dt * i) + traj_u.value(dt * (i+1)))/2;
                        
                        // calculate dynamics at midpoint
                        quadrotor_context_ptr->get_mutable_continuous_state().SetFromVector(state_value);
                        auto input_port_value = &quadrotor_context_ptr->FixInputPort(0, quadrotor->AllocateInputVector(quadrotor->get_input_port(0)));
                        input_port_value->systems::FixedInputPortValue::GetMutableVectorData<double>()->SetFromVector(input_value);
                        
                        Eigen::MatrixXd midpoint_derivative;
                        std::unique_ptr<systems::ContinuousState<double> > continuous_state(quadrotor->AllocateTimeDerivatives());
                        quadrotor->CalcTimeDerivatives(*quadrotor_context_ptr, continuous_state.get());
                        midpoint_derivative = continuous_state->CopyToVector();
                        
                        midpoint_error.col(i) = traj_x.value(dt * i) - (traj_x.value(dt * (i+1)) + dt * midpoint_derivative);
                    }
                    
                    // reshape/map matrix into vector
                    Map<VectorXd> error_vector(midpoint_error.data(), midpoint_error.size());
                    
                    // append to the end of the state files from before
                    output_file << error_vector.lpNorm<2>() << endl;
                    output_file << error_vector.lpNorm<Infinity>() << endl;
                    output_file.close();
                    
                    return 0;
                } */
                
                /* WRITE HEADER FILE */
                int writeHeaderFile(std::string filename, int trial, trajectories::PiecewisePolynomial<double> traj_x, trajectories::PiecewisePolynomial<double> traj_u, double time, double max_iter) {
                    
                    // open header file
                    std::string header_filename = output_folder + filename + "_" + std::to_string(trial) + ".txt";
                    output_file.open(header_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << header_filename << endl;
                        return -1;
                    }
                    
                    // write output
                    output_file << "Next lines: N, T, x0, xf, max_iter, time" << endl;
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
                    output_file << max_iter << endl;
                    output_file << time << endl;
                    
                    // calculate integration error
                    Eigen::MatrixXd midpoint_error(num_states, N-1);
                    for (int i = 0; i < N-1; i++) {
                        Eigen::VectorXd state_value = (traj_x.value(dt * i) + traj_x.value(dt * (i+1)))/2;
                        Eigen::VectorXd input_value = (traj_u.value(dt * i) + traj_u.value(dt * (i+1)))/2;
                        
                        // calculate dynamics at midpoint
                        quadrotor_context_ptr->get_mutable_continuous_state().SetFromVector(state_value);
                        auto input_port_value = &quadrotor_context_ptr->FixInputPort(0, quadrotor->AllocateInputVector(quadrotor->get_input_port(0)));
                        input_port_value->systems::FixedInputPortValue::GetMutableVectorData<double>()->SetFromVector(input_value);
                        
                        Eigen::MatrixXd midpoint_derivative;
                        std::unique_ptr<systems::ContinuousState<double> > continuous_state(quadrotor->AllocateTimeDerivatives());
                        quadrotor->CalcTimeDerivatives(*quadrotor_context_ptr, continuous_state.get());
                        midpoint_derivative = continuous_state->CopyToVector();
                        
                        midpoint_error.col(i) = traj_x.value(dt * i) - (traj_x.value(dt * (i+1)) + dt * midpoint_derivative);
                    }
                    
                    // reshape/map matrix into vector
                    Map<VectorXd> error_vector(midpoint_error.data(), midpoint_error.size());
                    
                    // append to the end of the state files from before
                    output_file << error_vector.lpNorm<2>() << endl;
                    output_file << error_vector.lpNorm<Infinity>() << endl;
                    output_file.close();
                    
                    return 0;
                }
                
                /*
                 void obstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                 
                 double quad_L = quadrotor->L();
                 
                 double right_side_x, right_side_y;
                 double left_side_x, left_side_y;
                 double rad_squared;
                 
                 for (int i = 0; i < obstacle_radii.size(); i++) {
                 // pre-compute
                 rad_squared = obstacle_radii[i] * obstacle_radii[i];
                 right_side_x = x[0] + quad_L * cos(x[2]) - obstacle_center_x[i];
                 right_side_y = x[1] + quad_L * sin(x[2]) - obstacle_center_y[i];
                 left_side_x = x[0] - quad_L * cos(x[2]) - obstacle_center_x[i];
                 left_side_y = x[1] - quad_L * sin(x[2]) - obstacle_center_y[i];
                 
                 // entries of d
                 g(3 * i) = rad_squared - (obstacle_center_x[i] - x[0]) * (obstacle_center_x[i] - x[0]) - (obstacle_center_y[i] - x[1]) * (obstacle_center_y[i] - x[1]);
                 g(3 * i + 1) = rad_squared - right_side_x * right_side_x - right_side_y * right_side_y;
                 g(3 * i + 2) = rad_squared - left_side_x * left_side_x - left_side_y * left_side_y;
                 
                 // entries of dd
                 dg_x(3 * i, 0) = -2 * (x[0] - obstacle_center_x[i]);
                 dg_x(3 * i, 1) = -2 * (x[1] - obstacle_center_y[i]);
                 
                 dg_x(3 * i + 1, 0) = -2 * right_side_x;
                 dg_x(3 * i + 1, 1) = -2 * right_side_y;
                 dg_x(3 * i + 1, 2) = -2 * right_side_x * (-quad_L * sin(x[3])) - 2 * right_side_y * (quad_L * cos(x[3]));
                 
                 dg_x(3 * i + 2, 0) = -2 * left_side_x;
                 dg_x(3 * i + 2, 1) = -2 * left_side_y;
                 dg_x(3 * i + 2, 2) = -2 * left_side_x * (-quad_L * sin(x[3])) - 2 * left_side_y * (quad_L * cos(x[3]));
                 }
                 } */
                
                void obstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                    
                    for (int i = 0; i < num_obstacles; i++) {
                        // entries of d
                        g(i) = 1 - (obstacle_center_x[i] - x[0]) * (obstacle_center_x[i] - x[0])/(obstacle_radii_x[i] * obstacle_radii_x[i]) - (obstacle_center_y[i] - x[1]) * (obstacle_center_y[i] - x[1])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                        
                        // entries of dd
                        dg_x(i, 0) = -2 * (x[0] - obstacle_center_x[i])/(obstacle_radii_x[i] * obstacle_radii_x[i]);
                        dg_x(i, 1) = -2 * (x[1] - obstacle_center_y[i])/(obstacle_radii_y[i] * obstacle_radii_y[i]);
                    }
                }
                
                void interpolatedObstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x1, Eigen::Ref<Eigen::VectorXd> u1, Eigen::Ref<Eigen::VectorXd> x2, Eigen::Ref<Eigen::VectorXd> u2, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x1, Eigen::Ref<Eigen::MatrixXd> dg_u1, Eigen::Ref<Eigen::MatrixXd> dg_x2, Eigen::Ref<Eigen::MatrixXd> dg_u2) {
                    
                    int num_alpha = 10;
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
                
                
                void solveSwingUpADMM(int trial, int max_iter) {
                    
                    cout << "num states: " << num_states << endl;
                    cout << "num inputs: " << num_inputs << endl;
                    cout << "num obstacles: " << num_obstacles << endl;
                    
                    // initialize solver
                    systems::trajectory_optimization::admm_solver::AdmmSolver solver = systems::trajectory_optimization::admm_solver::AdmmSolver(quadrotor, x0, xf, T, N, 1000);
                    
                    // initialize to a line between x0 and xf
                    Eigen::VectorXd y = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                    
                    if (LINEAR_WARM_START) {
                        for (int i = 0; i < N; i++) {
                            for (int ii = 0; ii < num_states; ii++) {
                                y[i * num_states + ii] = (1 - double(i)/(N-1)) * x0[ii] + double(i)/(N-1) * xf[ii];
                            }
                        }
                    }
                    
                    // set tolerances
                    solver.setRho1(500);
                    solver.setFeasibilityTolerance(1e-6);
                    
                    // use version with only center as a constraint
                    solver.addInequalityConstraintToAllKnotPoints(obstacleConstraints, num_obstacles, "obstacle constraints");
                    //solver.addInequalityConstraintToConsecutiveKnotPoints(interpolatedObstacleConstraints, num_obstacles * 10, "interpolated obstacle constraints");
                    
                    // add pitch constraint for consistency
                    solver.setStateUpperBound(state_upper_bound);
                    solver.setStateLowerBound(state_lower_bound);
                    solver.setInputLowerBound(input_lower_bound);
                    //solver.setInputUpperBound(input_upper_bound);
                    
                    // start timer
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    
                    // solve
                    solver.solve(y);
                    
                    // end timer
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    
                    cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    
                    trajectories::PiecewisePolynomial<double> xtraj_admm = solver.reconstructStateTrajectory();
                    trajectories::PiecewisePolynomial<double> utraj_admm = solver.reconstructInputTrajectory();
                    
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_admm_x", trial, xtraj_admm, num_states, elapsed_time.count(), max_iter);
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_admm_u", trial, utraj_admm, num_inputs, elapsed_time.count(), max_iter);
                    
                    //calculateIntegrationError(output_folder + "quadrotor_obstacles_admm_x", trial, xtraj_admm, utraj_admm);
                    
                    // write output to files
                    writeStateToFile("admm_x", trial, xtraj_admm);
                    writeInputToFile("admm_u", trial, utraj_admm);
                    
                    //Eigen::MatrixXd xtraj_admm = solver.getSolutionStateTrajectory();
                    //Eigen::MatrixXd utraj_admm = solver.getSolutionInputTrajectory();
                    
                    //writeOriginalTrajToFile("admm", trial, xtraj_admm, utraj_admm);
                    writeHeaderFile("admm_header", trial, xtraj_admm, utraj_admm, elapsed_time.count(), max_iter);
                    
                    //Eigen::MatrixXd xtraj_admm_original = solver.getSolutionStateTrajectory();
                    //Eigen::MatrixXd utraj_admm_original = solver.getSolutionInputTrajectory();
                    
                    // write obstacles to file
                    /*
                     ofstream output_admm;
                     output_admm.open(output_folder + "quadrotor_obstacles_admm_x_original.txt");
                     if (!output_admm.is_open()) {
                     cerr << "Problem opening admm original output file.\n";
                     }
                     output_admm << xtraj_admm_original << endl;
                     output_admm.close();
                     
                     output_admm.open(output_folder + "quadrotor_obstacles_admm_u_original.txt");
                     if (!output_admm.is_open()) {
                     cerr << "Problem opening admm original output file.\n";
                     }
                     output_admm << utraj_admm_original << endl;
                     output_admm.close(); */
                }
                
                void solveSwingUpWeightedADMM(int trial, int max_iter) {
                    
                    cout << "num states: " << num_states << endl;
                    cout << "num inputs: " << num_inputs << endl;
                    cout << "num obstacles: " << num_obstacles << endl;
                    
                    // initialize solver
                    systems::trajectory_optimization::admm_solver_weighted::AdmmSolverWeighted solver = systems::trajectory_optimization::admm_solver_weighted::AdmmSolverWeighted(quadrotor, x0, xf, T, N, 100);
                    //systems::trajectory_optimization::accel_admm_solver::AccelAdmmSolver solver = systems::trajectory_optimization::accel_admm_solver::AccelAdmmSolver(quadrotor, x0, xf, T, N, 1000);
                    
                    // initialize to a line between x0 and xf
                    Eigen::VectorXd y = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                    
                    if (LINEAR_WARM_START) {
                        for (int i = 0; i < N; i++) {
                            for (int ii = 0; ii < num_states; ii++) {
                                y[i * num_states + ii] = (1 - double(i)/(N-1)) * x0[ii] + double(i)/(N-1) * xf[ii];
                            }
                        }
                    }
                    
                    // set tolerances
                    solver.setRho1(500);
                    solver.setFeasibilityTolerance(1e-6);
                    
                    // use version with only center as a constraint
                    solver.addInequalityConstraintToAllKnotPoints(obstacleConstraints, num_obstacles, "obstacle constraints");
                    //solver.addInequalityConstraintToConsecutiveKnotPoints(interpolatedObstacleConstraints, num_obstacles * 10, "interpolated obstacle constraints");
                    
                    // add pitch constraint for consistency
                    solver.setStateUpperBound(state_upper_bound);
                    solver.setStateLowerBound(state_lower_bound);
                    solver.setInputLowerBound(input_lower_bound);
                    //solver.setInputUpperBound(input_upper_bound);
                    
                    // start timer
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    
                    // solve
                    solver.solve(y);
                    
                    // end timer
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    
                    cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    
                    trajectories::PiecewisePolynomial<double> xtraj_weighted_admm = solver.reconstructStateTrajectory();
                    trajectories::PiecewisePolynomial<double> utraj_weighted_admm = solver.reconstructInputTrajectory();
                    
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_admm_x", trial, xtraj_admm, num_states, elapsed_time.count(), max_iter);
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_admm_u", trial, utraj_admm, num_inputs, elapsed_time.count(), max_iter);
                    
                    //calculateIntegrationError(output_folder + "quadrotor_obstacles_admm_x", trial, xtraj_admm, utraj_admm);
                    
                    // write output to files
                    writeStateToFile("weighted_admm_x", trial, xtraj_weighted_admm);
                    writeInputToFile("weighted_admm_u", trial, utraj_weighted_admm);
                    writeHeaderFile("weighted_admm_header", trial, xtraj_weighted_admm, utraj_weighted_admm, elapsed_time.count(), max_iter);
                    
                    /*
                     Eigen::MatrixXd xtraj_admm_original = solver.getSolutionStateTrajectory();
                     Eigen::MatrixXd utraj_admm_original = solver.getSolutionInputTrajectory();
                     
                     // write obstacles to file
                     ofstream output_admm;
                     output_admm.open(output_folder + "quadrotor_obstacles_admm_x_original.txt");
                     if (!output_admm.is_open()) {
                     cerr << "Problem opening admm original output file.\n";
                     }
                     output_admm << xtraj_admm_original << endl;
                     output_admm.close();
                     
                     output_admm.open(output_folder + "quadrotor_obstacles_admm_u_original.txt");
                     if (!output_admm.is_open()) {
                     cerr << "Problem opening admm original output file.\n";
                     }
                     output_admm << utraj_admm_original << endl;
                     output_admm.close(); */
                }
                
                void solveSwingUpAccelADMM(int trial, int max_iter) {
                    
                    cout << "num states: " << num_states << endl;
                    cout << "num inputs: " << num_inputs << endl;
                    cout << "num obstacles: " << num_obstacles << endl;
                    
                    // initialize solver
                    //systems::trajectory_optimization::admm_solver_weighted::AdmmSolverWeighted solver = systems::trajectory_optimization::admm_solver_weighted::AdmmSolverWeighted(quadrotor, x0, xf, T, N, 1000);
                    systems::trajectory_optimization::accel_admm_solver::AccelAdmmSolver solver = systems::trajectory_optimization::accel_admm_solver::AccelAdmmSolver(quadrotor, x0, xf, T, N, 100);
                    
                    // initialize to a line between x0 and xf
                    Eigen::VectorXd y = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                    
                    if (LINEAR_WARM_START) {
                        for (int i = 0; i < N; i++) {
                            for (int ii = 0; ii < num_states; ii++) {
                                y[i * num_states + ii] = (1 - double(i)/(N-1)) * x0[ii] + double(i)/(N-1) * xf[ii];
                            }
                        }
                    }
                    
                    // set tolerances
                    solver.setRho1(500);
                    solver.setFeasibilityTolerance(1e-6);
                    
                    // use version with only center as a constraint
                    solver.addInequalityConstraintToAllKnotPoints(obstacleConstraints, num_obstacles, "obstacle constraints");
                    //solver.addInequalityConstraintToConsecutiveKnotPoints(interpolatedObstacleConstraints, num_obstacles * 10, "interpolated obstacle constraints");
                    
                    // add pitch constraint for consistency
                    solver.setStateUpperBound(state_upper_bound);
                    solver.setStateLowerBound(state_lower_bound);
                    solver.setInputLowerBound(input_lower_bound);
                    //solver.setInputUpperBound(input_upper_bound);
                    
                    // start timer
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    
                    // solve
                    solver.solve(y);
                    
                    // end timer
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    
                    cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    
                    trajectories::PiecewisePolynomial<double> xtraj_accel_admm = solver.reconstructStateTrajectory();
                    trajectories::PiecewisePolynomial<double> utraj_accel_admm = solver.reconstructInputTrajectory();
                    
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_admm_x", trial, xtraj_admm, num_states, elapsed_time.count(), max_iter);
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_admm_u", trial, utraj_admm, num_inputs, elapsed_time.count(), max_iter);
                    
                    //calculateIntegrationError(output_folder + "quadrotor_obstacles_admm_x", trial, xtraj_admm, utraj_admm);
                    
                    // write output to files
                    writeStateToFile("accel_admm_x", trial, xtraj_accel_admm);
                    writeInputToFile("accel_admm_u", trial, utraj_accel_admm);
                    writeHeaderFile("accel_admm_header", trial, xtraj_accel_admm, utraj_accel_admm, elapsed_time.count(), max_iter);
                    
                    /*
                     Eigen::MatrixXd xtraj_admm_original = solver.getSolutionStateTrajectory();
                     Eigen::MatrixXd utraj_admm_original = solver.getSolutionInputTrajectory();
                     
                     // write obstacles to file
                     ofstream output_admm;
                     output_admm.open(output_folder + "quadrotor_obstacles_admm_x_original.txt");
                     if (!output_admm.is_open()) {
                     cerr << "Problem opening admm original output file.\n";
                     }
                     output_admm << xtraj_admm_original << endl;
                     output_admm.close();
                     
                     output_admm.open(output_folder + "quadrotor_obstacles_admm_u_original.txt");
                     if (!output_admm.is_open()) {
                     cerr << "Problem opening admm original output file.\n";
                     }
                     output_admm << utraj_admm_original << endl;
                     output_admm.close(); */
                }
                
                void solveSwingUpAccelWeightedADMM(int trial, int max_iter) {
                    
                    cout << "num states: " << num_states << endl;
                    cout << "num inputs: " << num_inputs << endl;
                    cout << "num obstacles: " << num_obstacles << endl;
                    
                    // initialize solver
                    //systems::trajectory_optimization::admm_solver_weighted::AdmmSolverWeighted solver = systems::trajectory_optimization::admm_solver_weighted::AdmmSolverWeighted(quadrotor, x0, xf, T, N, 1000);
                    systems::trajectory_optimization::accel_weighted_admm_solver::AccelWeightedAdmmSolver solver = systems::trajectory_optimization::accel_weighted_admm_solver::AccelWeightedAdmmSolver(quadrotor, x0, xf, T, N, 100);
                    
                    // initialize to a line between x0 and xf
                    Eigen::VectorXd y = Eigen::VectorXd::Zero(N * (num_inputs + num_states));
                    
                    if (LINEAR_WARM_START) {
                        for (int i = 0; i < N; i++) {
                            for (int ii = 0; ii < num_states; ii++) {
                                y[i * num_states + ii] = (1 - double(i)/(N-1)) * x0[ii] + double(i)/(N-1) * xf[ii];
                            }
                        }
                    }
                    
                    // set tolerances
                    solver.setRho1(500);
                    solver.setFeasibilityTolerance(1e-6);
                    
                    // use version with only center as a constraint
                    solver.addInequalityConstraintToAllKnotPoints(obstacleConstraints, num_obstacles, "obstacle constraints");
                    //solver.addInequalityConstraintToConsecutiveKnotPoints(interpolatedObstacleConstraints, num_obstacles * 10, "interpolated obstacle constraints");
                    
                    // add pitch constraint for consistency
                    solver.setStateUpperBound(state_upper_bound);
                    solver.setStateLowerBound(state_lower_bound);
                    solver.setInputLowerBound(input_lower_bound);
                    //solver.setInputUpperBound(input_upper_bound);
                    
                    // start timer
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    
                    // solve
                    solver.solve(y);
                    
                    // end timer
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    
                    cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    
                    trajectories::PiecewisePolynomial<double> xtraj_accel_admm = solver.reconstructStateTrajectory();
                    trajectories::PiecewisePolynomial<double> utraj_accel_admm = solver.reconstructInputTrajectory();
                    
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_admm_x", trial, xtraj_admm, num_states, elapsed_time.count(), max_iter);
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_admm_u", trial, utraj_admm, num_inputs, elapsed_time.count(), max_iter);
                    
                    //calculateIntegrationError(output_folder + "quadrotor_obstacles_admm_x", trial, xtraj_admm, utraj_admm);
                    
                    // write output to files
                    writeStateToFile("accel_admm_x", trial, xtraj_accel_admm);
                    writeInputToFile("accel_admm_u", trial, utraj_accel_admm);
                    writeHeaderFile("accel_admm_header", trial, xtraj_accel_admm, utraj_accel_admm, elapsed_time.count(), max_iter);
                    
                    /*
                     Eigen::MatrixXd xtraj_admm_original = solver.getSolutionStateTrajectory();
                     Eigen::MatrixXd utraj_admm_original = solver.getSolutionInputTrajectory();
                     
                     // write obstacles to file
                     ofstream output_admm;
                     output_admm.open(output_folder + "quadrotor_obstacles_admm_x_original.txt");
                     if (!output_admm.is_open()) {
                     cerr << "Problem opening admm original output file.\n";
                     }
                     output_admm << xtraj_admm_original << endl;
                     output_admm.close();
                     
                     output_admm.open(output_folder + "quadrotor_obstacles_admm_u_original.txt");
                     if (!output_admm.is_open()) {
                     cerr << "Problem opening admm original output file.\n";
                     }
                     output_admm << utraj_admm_original << endl;
                     output_admm.close(); */
                }
                
                solvers::SolutionResult solveSwingUpIPOPT(int trial, int max_iter) {
                    systems::trajectory_optimization::MidpointTranscription dircol(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    //systems::trajectory_optimization::DirectCollocation dircol(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    
                    dircol.AddEqualTimeIntervalsConstraints();
                    
                    auto u = dircol.input();
                    dircol.AddConstraintToAllKnotPoints(input_lower_bound <= u);
                    //dircol.AddConstraintToAllKnotPoints(u <= input_upper_bound);
                    
                    dircol.AddLinearConstraint(dircol.initial_state() == x0);
                    dircol.AddLinearConstraint(dircol.final_state() == xf);
                    
                    //const double R = 1;  // Cost on input "effort".
                    const Eigen::Matrix4d R = Eigen::MatrixXd::Identity(4, 4);
                    dircol.AddRunningCost(u.dot(R * u));
                    //dircol.AddRunningCost(u.dot() * u);
                    
                    const double timespan_init = T;
                    
                    auto traj_init_x = PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, x0});
                    if (LINEAR_WARM_START) {
                        traj_init_x = PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, xf});
                    }
                    dircol.SetInitialTrajectory(PiecewisePolynomialType(), traj_init_x);
                    
                    // set tolerance low so that it does not interfere with maximum iteration comparison?
                    dircol.SetSolverOption(solvers::IpoptSolver::id(), "tol", 1e-10);
                    dircol.SetSolverOption(solvers::IpoptSolver::id(), "constr_viol_tol", 1e-10);
                    dircol.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_tol", 1e-10);
                    dircol.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_constr_viol_tol", 1e-10);
                    
                    // set maximum iterations
                    //dircol.SetSolverOption(solvers::IpoptSolver::id(), "max_iter", max_iter);
                    
                    // verbose?
                    dircol.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 5);
                    
                    // add obstacle constraints!
                    auto x = dircol.state();
                    //for (int i = 0; i < obstacle_radii.size(); i++) {
                    
                    for (int i = 0; i < num_obstacles; i++) {
                        dircol.AddConstraintToAllKnotPoints((x(0) - obstacle_center_x(i)) * (x(0) - obstacle_center_x(i))/(obstacle_radii_x(i) * obstacle_radii_x(i)) + (x(1) - obstacle_center_y(i)) * (x(1) - obstacle_center_y(i))/(obstacle_radii_y(i) * obstacle_radii_y(i)) >= 1);
                    }
                    
                    //dircol.AddInterpolatedObstacleConstraintToAllPoints(obstacle_center_x, obstacle_center_y, obstacle_radii, 5);
                    //}
                    
                    // constrain angle?
                    dircol.AddConstraintToAllKnotPoints(x <= state_upper_bound);
                    dircol.AddConstraintToAllKnotPoints(x >= state_lower_bound);
                    
                    solvers::IpoptSolver solver;
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    solvers::SolutionResult result = solver.Solve(dircol);
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    std::cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    std::cout << "Solution result:" << result << "\n";
                    
                    trajectories::PiecewisePolynomial<double> xtraj_ipopt = dircol.ReconstructStateTrajectory();
                    trajectories::PiecewisePolynomial<double> utraj_ipopt = dircol.ReconstructInputTrajectory();
                    
                    // write output to files
                    writeStateToFile("ipopt_x", trial, xtraj_ipopt);
                    writeInputToFile("ipopt_u", trial, utraj_ipopt);
                    writeHeaderFile("ipopt_header", trial, xtraj_ipopt, utraj_ipopt, elapsed_time.count(), max_iter);
                    
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_ipopt_x", trial, xtraj_ipopt, num_states, elapsed_time.count(), max_iter);
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_ipopt_u", trial, utraj_ipopt, num_inputs, elapsed_time.count(), max_iter);
                    
                    //calculateIntegrationError(output_folder + "quadrotor_obstacles_ipopt_x", trial, xtraj_ipopt, utraj_ipopt);
                    
                    return result;
                }
                
                solvers::SolutionResult solveSwingUpSNOPT(int trial, int max_iter) {
                    cout << "\n-- Solving with SNOPT --" << endl;
                    systems::trajectory_optimization::MidpointTranscription dircol(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    //systems::trajectory_optimization::DirectCollocation dircol(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    
                    dircol.AddEqualTimeIntervalsConstraints();
                    
                    auto u = dircol.input();
                    //dircol.AddConstraintToAllKnotPoints(-torqueLimit <= u(0));
                    dircol.AddConstraintToAllKnotPoints(u >= input_lower_bound);
                    //dircol.AddConstraintToAllKnotPoints(u <= input_upper_bound);
                    
                    dircol.AddLinearConstraint(dircol.initial_state() == x0);
                    dircol.AddLinearConstraint(dircol.final_state() == xf);
                    
                    //const double R = 1;  // Cost on input "effort".
                    const Eigen::Matrix4d R = Eigen::MatrixXd::Identity(4, 4);
                    dircol.AddRunningCost(u.dot(R * u));
                    //dircol.AddRunningCost(u * u);
                    
                    // add obstacle constraints!
                    auto x = dircol.state();
                    //for (int i = 0; i < obstacle_radii.size(); i++) {
                    for (int i = 0; i < num_obstacles; i++) {
                        dircol.AddConstraintToAllKnotPoints((x(0) - obstacle_center_x(i)) * (x(0) - obstacle_center_x(i))/(obstacle_radii_x(i) * obstacle_radii_x(i)) + (x(1) - obstacle_center_y(i)) * (x(1) - obstacle_center_y(i))/(obstacle_radii_y(i) * obstacle_radii_y(i)) >= 1);
                    }
                    
                    //dircol.AddInterpolatedObstacleConstraintToAllPoints(obstacle_center_x, obstacle_center_y, obstacle_radii, 5);
                    
                    // THRUST SHOULD BE POSITIVE
                    
                    // constrain angle with upper/lower bounds
                    dircol.AddConstraintToAllKnotPoints(x <= state_upper_bound);
                    dircol.AddConstraintToAllKnotPoints(x >= state_lower_bound);
                    
                    const double timespan_init = T;
                    
                    auto traj_init_x = PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, x0});
                    if (LINEAR_WARM_START) {
                        traj_init_x = PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, xf});
                    }
                    dircol.SetInitialTrajectory(PiecewisePolynomialType(), traj_init_x);
                    
                    // set tolerance to be very small
                    dircol.SetSolverOption(solvers::SnoptSolver::id(), "Major feasibility tolerance", 1e-8);
                    dircol.SetSolverOption(solvers::SnoptSolver::id(), "Major optimality tolerance", 1e-4);
                    
                    // set tolerance
                    dircol.SetSolverOption(solvers::SnoptSolver::id(), "Iterations limit", 100000);
                    
                    // verbose?
                    //dircol.SetSolverOption(solvers::SnoptSolver::id(), "Major print level", 2);
                    //dircol.SetSolverOption(solvers::SnoptSolver::id(), "Print file", "snopt_output");
                    const std::string print_file = output_folder + "snopt" + std::to_string(trial) + ".out";
                    cout << "Should be printing to " << print_file << endl;
                    dircol.SetSolverOption(solvers::SnoptSolver::id(), "Print file", print_file);
                    
                    // solve
                    solvers::SnoptSolver solver;
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    solvers::SolutionResult result = solver.Solve(dircol);
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    std::cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    std::cout << "Solution result:" << result << "\n";
                    
                    trajectories::PiecewisePolynomial<double> xtraj_snopt = dircol.ReconstructStateTrajectory();
                    trajectories::PiecewisePolynomial<double> utraj_snopt = dircol.ReconstructInputTrajectory();
                    
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_snopt_x", trial, xtraj_snopt, num_states, elapsed_time.count(), max_iter);
                    //writeTrajToFileTol(output_folder + "quadrotor_obstacles_snopt_u", trial, utraj_snopt, num_inputs, elapsed_time.count(), max_iter);
                    
                    //calculateIntegrationError(output_folder + "quadrotor_obstacles_snopt_x", trial, xtraj_snopt, utraj_snopt);
                    
                    writeStateToFile("snopt_x", trial, xtraj_snopt);
                    writeInputToFile("snopt_u", trial, utraj_snopt);
                    writeHeaderFile("snopt_header", trial, xtraj_snopt, utraj_snopt, elapsed_time.count(), max_iter);
                    
                    return result;
                }
                
                int do_main(int argc, char* argv[]) {
                    // repeat randomization and solution for multiple trials
                    int num_trials = 2;
                    std::vector<solvers::SolutionResult> ipopt_results(num_trials);
                    std::vector<solvers::SolutionResult> snopt_results(num_trials);
                    
                    // initialize some number of trees
                    num_obstacles = 6;
                    obstacle_center_x = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_center_y = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_radii_x = Eigen::VectorXd::Zero(num_obstacles);
                    obstacle_radii_y = Eigen::VectorXd::Zero(num_obstacles);
                    
                    // set state bounds
                    state_upper_bound << 100, 100, 100, 100, 0.2, 100, 100, 100, 100, 100, 100, 100;
                    state_lower_bound << -100, -100, -100, -100, -0.2, -100, -100, -100, -100, -100, -100, -100;
                    input_lower_bound << 0, 0, 0, 0;
                    input_upper_bound << 10, 10, 10, 10;
                    
                    // make random generators for the trees
                    std::default_random_engine generator;
                    std::uniform_real_distribution<double> x_distribution(0.0, 6.0);
                    std::uniform_real_distribution<double> y_distribution(0.0, 6.0);
                    std::normal_distribution<double> radii_x_distribution(0.5, 0.1);
                    std::normal_distribution<double> radii_y_distribution(0.5, 0.1);
                    
                    // repeat for some number of trials
                    for (int trial = 0; trial < num_trials; trial++) {
                        
                        // put obstacles at those locations
                        int obs_count = 0;
                        while(obs_count < num_obstacles) {
                            // sample obstacle
                            obstacle_center_x[obs_count] = x_distribution(generator);
                            obstacle_center_y[obs_count] = y_distribution(generator);
                            obstacle_radii_x[obs_count] = radii_x_distribution(generator);
                            obstacle_radii_y[obs_count] = radii_y_distribution(generator);
                            
                            // check if start point is not in obstacle
                            bool x0_feasible = (x0[0] - obstacle_center_x[obs_count]) * (x0[0] - obstacle_center_x[obs_count])/(obstacle_radii_x[obs_count] * obstacle_radii_x[obs_count]) + (x0[1] - obstacle_center_y[obs_count]) * (x0[1] - obstacle_center_y[obs_count])/(obstacle_radii_y[obs_count] * obstacle_radii_y[obs_count]) > 1;
                            
                            // check if end point is not in obstacle
                            bool xf_feasible = (xf[0] - obstacle_center_x[obs_count]) * (xf[0] - obstacle_center_x[obs_count])/(obstacle_radii_x[obs_count] * obstacle_radii_x[obs_count]) + (xf[1] - obstacle_center_y[obs_count]) * (xf[1] - obstacle_center_y[obs_count])/(obstacle_radii_y[obs_count] * obstacle_radii_y[obs_count]) > 1;
                            
                            // if all are clear, accept obstacle
                            if (x0_feasible & xf_feasible) {
                                obs_count = obs_count + 1;
                            }
                            
                        }
                        
                        // gather statistics
                        
                        // make flag to check realisticness of solution
                        
                        // check if SNOPT and IPOPT perform consistently under certain settings of obstacles (intersection?)
                        
                        int max_iter = 400; // currently does nothing
                        
                        // solve!
                        solveSwingUpADMM(trial, max_iter);
                        snopt_results[trial] = solveSwingUpSNOPT(trial, max_iter);
                        ipopt_results[trial] = solveSwingUpIPOPT(trial, max_iter);
                        
                        // write obstacles to file
                        ofstream output_obs;
                        output_obs.open(output_folder + "obstacles_" + std::to_string(trial) + ".txt");
                        if (!output_obs.is_open()) {
                            cerr << "Problem opening obstacle output file.\n";
                        }
                        output_obs << obstacle_center_x.transpose() << endl;
                        output_obs << obstacle_center_y.transpose() << endl;
                        output_obs << obstacle_radii_x.transpose() << endl;
                        output_obs << obstacle_radii_y.transpose() << endl;
                        output_obs.close();
                    }
                    
                    cout << "\n\nIPOPT Results:\n";
                    for (int trial = 0; trial < num_trials; trial++) {
                        cout << ipopt_results[trial] << endl;
                    }
                    cout << "\n\nSNOPT Results:\n";
                    for (int trial = 0; trial < num_trials; trial++) {
                        cout << snopt_results[trial] << endl;
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





