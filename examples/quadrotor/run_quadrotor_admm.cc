
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
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"

#define LINEAR_WARM_START 0

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
                const Eigen::VectorXd xf = (Eigen::VectorXd(12) << 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
                
                Eigen::VectorXd state_upper_bound(12);
                Eigen::VectorXd state_lower_bound(12);
                
                double T = 10.0;
                int N = 40;
                double dt = T/N;
                
                int num_states = quadrotor_context_ptr->get_num_total_states();
                int num_inputs = quadrotor->get_input_port(0).size();
                
                // prepare output file writer and control input for dynamics integration!
                ofstream output_file;
                trajectories::PiecewisePolynomial<double> dynamics_tau;
                
                // MAKE A LIST OF CYLINDRICAL OBSTACLES TO AVOID
                //int num_obstacles = 6;
                Eigen::VectorXd obstacle_center_x(6);
                Eigen::VectorXd obstacle_center_y(6);
                Eigen::VectorXd obstacle_radii(6);
                
                // function to calculate integration error of a solution trajectory and write to file
                void calculateIntegrationError(std::string filename, int trial, trajectories::PiecewisePolynomial<double> traj_x, trajectories::PiecewisePolynomial<double> traj_u) {
                    
                    // calculate error
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
                    std::string traj_filename = filename + "_" + std::to_string(trial) + "_params.txt";
                    output_file.open(traj_filename, std::ios_base::app);
                    output_file << error_vector.lpNorm<2>() << endl;
                    output_file << error_vector.lpNorm<Infinity>() << endl;
                    output_file.close();
                }
                
                //assigns values to obstacles and state lower/upper bounds
                void initialize() {
                    obstacle_center_x << 1.5, 1.5, 1.5, 3.5, 3.5, 3.5;
                    obstacle_center_y << 0.5, 2.5, 4.5, 1.5, 3.5, 5.5;
                    obstacle_radii << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
                    
                    state_upper_bound << 100, 100, 100, 100, 0.2, 100, 100, 100, 100, 100, 100, 100;
                    state_lower_bound << -100, -100, -100, -100, -0.2, -100, -100, -100, -100, -100, -100, -100;
                    
                    //state_upper_bound << 100, 100, 100, 100;
                    //state_lower_bound << -100, -100, -100, -100;
                }
                
                int writeTrajToFileTol(std::string filename, int trial, trajectories::PiecewisePolynomial<double> traj, int m, double time, double max_iter) {
                    // write trajectory to file
                    std::string traj_filename = filename + "_" + std::to_string(trial) + ".txt";
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << endl;
                        return -1;
                    }
                    
                    for (int i = 0; i < N; i++) {
                        Eigen::VectorXd x = traj.value(i * T/(N-1));
                        // write time
                        output_file << i * T/(N-1) << '\t';
                        // write all state or input values
                        for (int ii = 0; ii < m; ii++) {
                            output_file << x[ii] << '\t';
                        }
                        output_file << endl;
                    }
                    output_file.close();
                    
                    // write "header" file with info?
                    std::string header_filename = filename + "_" + std::to_string(trial) + "_params.txt";
                    output_file.open(header_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << header_filename << endl;
                        return -1;
                    }
                    output_file << "Next lines: N, T, x0, xf, max_iter, time" << endl;
                    output_file << N << endl;
                    output_file << T << endl;
                    for (int ii = 0; ii < m; ii++) {
                        output_file << x0[ii] << '\t';
                    }
                    output_file << endl;
                    for (int ii = 0; ii < m; ii++) {
                        output_file << xf[ii] << '\t';
                    }
                    output_file << endl;
                    output_file << max_iter << endl;
                    output_file << time << endl;
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
                 } */ /*
                       void obstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                       
                       for (int i = 0; i < obstacle_radii.size(); i++) {
                       // entries of d
                       g(i) = obstacle_radii[i] * obstacle_radii[i] - (obstacle_center_x[i] - x[0]) * (obstacle_center_x[i] - x[0]) - (obstacle_center_y[i] - x[1]) * (obstacle_center_y[i] - x[1]);
                       
                       // entries of dd
                       dg_x(i, 0) = -2 * (x[0] - obstacle_center_x[i]);
                       dg_x(i, 1) = -2 * (x[1] - obstacle_center_y[i]);
                       }
                       } */
                
                void solveSwingUpADMM(int trial, int max_iter) {

                    
                    // initialize solver
                    systems::trajectory_optimization::AdmmSolver solver = systems::trajectory_optimization::AdmmSolver(quadrotor, x0, xf, T, N, 1000);
                    
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
                    solver.setFeasibilityTolerance(1e-8);
                    
                    // use version with only center as a constraint
                    //solver.addInequalityConstraintToAllKnotPoints(obstacleConstraints, obstacle_radii.size(), "obstacle constraints");
                    
                    // add pitch constraint for consistency
                    solver.setStateUpperBound(state_upper_bound);
                    solver.setStateLowerBound(state_lower_bound);
                    
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
                    
                    writeTrajToFileTol("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_admm_x", trial, xtraj_admm, num_states, elapsed_time.count(), max_iter);
                    writeTrajToFileTol("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_admm_u", trial, utraj_admm, num_inputs, elapsed_time.count(), max_iter);
                    
                    calculateIntegrationError("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_admm_x", trial, xtraj_admm, utraj_admm);
                }
                
                solvers::SolutionResult solveSwingUpIPOPT(int trial, int max_iter) {
                    //systems::trajectory_optimization::MidpointTranscription dircol(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    systems::trajectory_optimization::DirectCollocation dircol(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    
                    dircol.AddEqualTimeIntervalsConstraints();
                    
                    auto u = dircol.input();
                    
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
                    //for (int i = 0; i < num_obstacles; i++) {
                    //    dircol.AddConstraintToAllKnotPoints((x(0) - obstacle_center_x(i)) * (x(0) - obstacle_center_x(i)) + (x(1) - obstacle_center_y(i)) * (x(1) - obstacle_center_y(i)) >= obstacle_radii(i) * obstacle_radii(i));
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
                    
                    writeTrajToFileTol("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_ipopt_x", trial, xtraj_ipopt, num_states, elapsed_time.count(), max_iter);
                    writeTrajToFileTol("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_ipopt_u", trial, utraj_ipopt, num_inputs, elapsed_time.count(), max_iter);
                    
                    calculateIntegrationError("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_ipopt_x", trial, xtraj_ipopt, utraj_ipopt);
                    
                    return result;
                }
                
                solvers::SolutionResult solveSwingUpSNOPT(int trial, int max_iter) {
                    cout << "\n-- Solving with SNOPT --" << endl;
                    //systems::trajectory_optimization::MidpointTranscription dircol(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    systems::trajectory_optimization::MidpointTranscription dircol(quadrotor, *quadrotor_context_ptr, N, T/N, T/N);
                    
                    dircol.AddEqualTimeIntervalsConstraints();
                    
                    auto u = dircol.input();
                    //dircol.AddConstraintToAllKnotPoints(-torqueLimit <= u(0));
                    //dircol.AddConstraintToAllKnotPoints(u(0) <= torqueLimit);
                    
                    dircol.AddLinearConstraint(dircol.initial_state() == x0);
                    dircol.AddLinearConstraint(dircol.final_state() == xf);
                    
                    //const double R = 1;  // Cost on input "effort".
                    const Eigen::Matrix4d R = Eigen::MatrixXd::Identity(4, 4);
                    dircol.AddRunningCost(u.dot(R * u));
                    //dircol.AddRunningCost(u * u);
                    
                    // add obstacle constraints!
                    auto x = dircol.state();
                    //for (int i = 0; i < obstacle_radii.size(); i++) {
                    //for (int i = 0; i < num_obstacles; i++) {
                    //    dircol.AddConstraintToAllKnotPoints((x(0) - obstacle_center_x(i)) * (x(0) - obstacle_center_x(i)) + (x(1) - obstacle_center_y(i)) * (x(1) - obstacle_center_y(i)) >= obstacle_radii(i) * obstacle_radii(i));
                    //}
                    
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
                    dircol.SetSolverOption(solvers::SnoptSolver::id(), "Major feasibility tolerance", 1e-10);
                    dircol.SetSolverOption(solvers::SnoptSolver::id(), "Major optimality tolerance", 1e-10);
                    
                    // set tolerance
                    //dircol.SetSolverOption(solvers::SnoptSolver::id(), "Major iterations limit", max_iter);
                    
                    // verbose?
                    //dircol.SetSolverOption(solvers::SnoptSolver::id(), "Major print level", 2);
                    //dircol.SetSolverOption(solvers::SnoptSolver::id(), "Print file", "snopt_output");
                    const std::string print_file = "/Users/irina/Documents/drake/examples/quadrotor/output/snopt" + std::to_string(trial) + ".out";
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
                    
                    writeTrajToFileTol("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_snopt_x", trial, xtraj_snopt, num_states, elapsed_time.count(), max_iter);
                    writeTrajToFileTol("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_snopt_u", trial, utraj_snopt, num_inputs, elapsed_time.count(), max_iter);
                    
                    calculateIntegrationError("/Users/irina/Documents/drake/examples/quadrotor/output/quadrotor_obstacles_snopt_x", trial, xtraj_snopt, utraj_snopt);
                    
                    return result;
                }
                
                int do_main(int argc, char* argv[]) {
                    initialize();
                    
                    int max_iter = 400;
                    std::vector<solvers::SolutionResult> ipopt_results(1);
                    std::vector<solvers::SolutionResult> snopt_results(1);
                    
                    solveSwingUpADMM(0, max_iter);
                    ipopt_results[0] = solveSwingUpIPOPT(0, max_iter);
                    snopt_results[0] = solveSwingUpSNOPT(0, max_iter);
                    
                    return 0;
                }
                
            }  // namespace
        }  // namespace quadrotor
    }  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    return drake::examples::quadrotor::do_main(argc, argv);
}




