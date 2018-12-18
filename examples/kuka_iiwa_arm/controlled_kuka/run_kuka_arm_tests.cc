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

DEFINE_double(simulation_sec, 1, "Number of seconds to simulate.");

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
                const Eigen::VectorXd x0 = (Eigen::VectorXd(12) << 0, -0.683, 0, 1.77, 0, 0.88, -1.57, 0, 0, 0, 0, 0, 0, 0).finished();
                const Eigen::VectorXd xf = (Eigen::VectorXd(12) << 0, 0, 0, -pi/4.0, 0, pi/4.0, pi/2.0, 0, 0, 0, 0, 0, 0, 0).finished();
                int num_states;
                int num_inputs;
                
                // define time and number of points
                int N = 10;
                double T = 10.0;
                double dt = T/N;
                
                // prepare trajectories
                trajectories::PiecewisePolynomial<double> xtraj_ipopt;
                trajectories::PiecewisePolynomial<double> utraj_ipopt;
                trajectories::PiecewisePolynomial<double> xtraj_snopt;
                trajectories::PiecewisePolynomial<double> utraj_snopt;
                
                // for writing files
                std::ofstream output_file;
                std::string output_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/basic/";
                
                
                //=============================================================================//
                
                
                /* WRITE STATE TO FILE */
                int writeStateToFile(std::string filename, int trial, trajectories::PiecewisePolynomial<double> traj) {
                    // filename
                    std::string traj_filename = output_folder + filename + "_" + std::to_string(trial) + ".txt";
                    
                    // open output file
                    output_file.open(traj_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << std::endl;
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
                        output_file << std::endl;
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
                        std::cerr << "Problem opening file at " << traj_filename << std::endl;
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
                        output_file << std::endl;
                    }
                    output_file.close();
                    
                    return 0;
                }
                
                /* WRITE HEADER FILE */
                int writeHeaderFile(std::string filename, int trial, trajectories::PiecewisePolynomial<double> traj_x, trajectories::PiecewisePolynomial<double> traj_u, double time, double max_iter) {
                    
                    // open header file
                    std::string header_filename = output_folder + filename + "_" + std::to_string(trial) + ".txt";
                    output_file.open(header_filename);
                    if (!output_file.is_open()) {
                        std::cerr << "Problem opening file at " << header_filename << std::endl;
                        return -1;
                    }
                    
                    // write output
                    output_file << "Next lines: N, T, x0, xf, max_iter, time" << std::endl;
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
                    output_file << max_iter << std::endl;
                    output_file << time << std::endl;
                    
                    // calculate integration error
                    /*
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
                    Map<Eigen::VectorXd> error_vector(midpoint_error.data(), midpoint_error.size());
                    
                    // append to the end of the state files from before
                    output_file << error_vector.lpNorm<2>() << std::endl;
                    output_file << error_vector.lpNorm<Infinity>() << std::endl; */
                    output_file.close();
                    
                    return 0;
                }
                
                /* Solve with IPOPT */
                solvers::SolutionResult solveIPOPT(drake::systems::RigidBodyPlant<double>* plant) {
                    systems::trajectory_optimization::MidpointTranscription traj_opt(plant, *plant->CreateDefaultContext(), N, dt, dt);
                    
                    // get state and input placeholders
                    auto u = traj_opt.input();
                    auto x = traj_opt.state();
                    
                    // state and input bounds
                    Eigen::VectorXd state_lower_bound = Eigen::VectorXd::Ones(num_states) * -pi/2;
                    Eigen::VectorXd state_upper_bound = Eigen::VectorXd::Ones(num_states) * pi/2;
                    Eigen::VectorXd input_lower_bound = Eigen::VectorXd::Ones(num_inputs) * -200;
                    Eigen::VectorXd input_upper_bound = Eigen::VectorXd::Ones(num_inputs) * 200;
                    
                    // add input limits to problem
                    traj_opt.AddConstraintToAllKnotPoints(u >= input_lower_bound);
                    traj_opt.AddConstraintToAllKnotPoints(u <= input_upper_bound);
                    
                    // add joint limits to problem
                    traj_opt.AddConstraintToAllKnotPoints(x <= state_upper_bound);
                    traj_opt.AddConstraintToAllKnotPoints(x >= state_lower_bound);
                    
                    // matrices for running and final costs
                    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(num_states, num_states);
                    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(num_states, num_states) * 1000;
                    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(num_inputs, num_inputs) * 0.0001;
                    
                    // add constraints to problem
                    traj_opt.AddRunningCost((x - xf).dot(Qf * (x - xf)) + u.dot(R * u));
                    traj_opt.AddFinalCost((x - xf).dot(Q * (x - xf)));
                    
                    // ipopt settings?
                    traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "tol", 1e-4);
                    traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "constr_viol_tol", 1e-4);
                    traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_tol", 1e-4);
                    traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "acceptable_constr_viol_tol", 1e-4);
                    traj_opt.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 5);
                    
                    // solve and time solution
                    solvers::IpoptSolver solver;
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    solvers::SolutionResult result = solver.Solve(traj_opt);
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    std::cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    std::cout << "Solution result:" << result << "\n";
                    
                    // make trajectories
                    xtraj_ipopt = traj_opt.ReconstructStateTrajectory();
                    utraj_ipopt = traj_opt.ReconstructInputTrajectory();
                    
                    // write output to files
                    writeStateToFile("ipopt_x", 0, xtraj_ipopt);
                    writeInputToFile("ipopt_u", 0, utraj_ipopt);
                    writeHeaderFile("ipopt_header", 0, xtraj_ipopt, utraj_ipopt, elapsed_time.count(), -1);
                    
                    return result;
                }
                
                /* Solve with IPOPT */
                solvers::SolutionResult solveSNOPT(drake::systems::RigidBodyPlant<double>* plant) {
                    systems::trajectory_optimization::MidpointTranscription traj_opt(plant, *plant->CreateDefaultContext(), N, dt, dt);
                    
                    // get state and input placeholders
                    auto u = traj_opt.input();
                    auto x = traj_opt.state();
                    
                    // state and input bounds
                    Eigen::VectorXd state_lower_bound = Eigen::VectorXd::Ones(num_states) * -pi/2;
                    Eigen::VectorXd state_upper_bound = Eigen::VectorXd::Ones(num_states) * pi/2;
                    Eigen::VectorXd input_lower_bound = Eigen::VectorXd::Ones(num_inputs) * -200;
                    Eigen::VectorXd input_upper_bound = Eigen::VectorXd::Ones(num_inputs) * 200;
                    
                    // add input limits to problem
                    traj_opt.AddConstraintToAllKnotPoints(u >= input_lower_bound);
                    traj_opt.AddConstraintToAllKnotPoints(u <= input_upper_bound);
                    
                    // add joint limits to problem
                    traj_opt.AddConstraintToAllKnotPoints(x <= state_upper_bound);
                    traj_opt.AddConstraintToAllKnotPoints(x >= state_lower_bound);
                    
                    // matrices for running and final costs
                    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(num_states, num_states);
                    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(num_states, num_states) * 1000;
                    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(num_inputs, num_inputs) * 0.0001;
                    
                    // add constraints to problem
                    traj_opt.AddRunningCost((x - xf).dot(Qf * (x - xf)) + u.dot(R * u));
                    traj_opt.AddFinalCost((x - xf).dot(Q * (x - xf)));
                    
                    // set SNOPT options
                    traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major feasibility tolerance", 1e-4);
                    traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Major optimality tolerance", 1e-4);
                    traj_opt.SetSolverOption(solvers::SnoptSolver::id(), "Iterations limit", 100000);
                    
                    // solve and time solution
                    solvers::SnoptSolver solver;
                    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
                    solvers::SolutionResult result = solver.Solve(traj_opt);
                    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                    
                    std::chrono::duration<double> elapsed_time = (end_time - start_time);
                    std::cout << "Finished! Runtime = " << elapsed_time.count() << " sec. \n";
                    std::cout << "Solution result:" << result << "\n";
                    
                    // make trajectories
                    xtraj_snopt = traj_opt.ReconstructStateTrajectory();
                    utraj_snopt = traj_opt.ReconstructInputTrajectory();
                    
                    // write output to files
                    writeStateToFile("snopt_x", 0, xtraj_snopt);
                    writeInputToFile("snopt_u", 0, utraj_snopt);
                    writeHeaderFile("snopt_header", 0, xtraj_snopt, utraj_snopt, elapsed_time.count(), 100000);
                    
                    return result;
                }
                
                
                int DoMain() {
                    
                    // make plant
                    DRAKE_DEMAND(FLAGS_simulation_sec > 0);
                    
                    // create tree, plant, builder
                    auto tree = std::make_unique<RigidBodyTree<double>>();
                    CreateTreedFromFixedModelAtPose(FindResourceOrThrow(kUrdfPath), tree.get());
                    drake::lcm::DrakeLcm lcm;
                    SimDiagramBuilder<double> builder;
                    drake::systems::RigidBodyPlant<double>* plant = builder.AddPlant(std::move(tree));
                    builder.AddVisualizer(&lcm);
                    
                    num_states = plant->get_num_positions() + plant->get_num_velocities();
                    num_inputs = plant->get_num_actuators();
                    
                    // check constants
                    std::cout << "num bodies: " << plant->get_num_bodies() << std::endl;
                    std::cout << "num positions: " << plant->get_num_positions() << std::endl;
                    std::cout << "num actuators: " << plant->get_num_actuators() << std::endl;
                    
                    solveIPOPT(plant);
                    solveSNOPT(plant);
                    
                    /*
                    std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
                    systems::Simulator<double> simulator(*diagram);
                    simulator.Initialize();
                    simulator.set_target_realtime_rate(1.0); */
                    
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
