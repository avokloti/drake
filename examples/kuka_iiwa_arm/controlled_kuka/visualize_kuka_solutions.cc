//
//  visualize_kuka_solutions.cpp
//  
//
//  Created by Irina Tolkova on 12/18/18.
//

#include <stdio.h>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/manipulation/util/sim_diagram_builder.h"
#include "drake/multibody/parsers/urdf_parser.h"

DEFINE_double(simulation_sec, 7, "Number of seconds to simulate.");

using drake::geometry::SceneGraph;
using drake::lcm::DrakeLcm;

// from rigidbody file:
using drake::manipulation::util::SimDiagramBuilder;
using drake::trajectories::PiecewisePolynomial;

namespace drake {
    namespace examples {
        namespace kuka_iiwa_arm {
            namespace {
                
                const char kUrdfPath[] =
                "drake/manipulation/models/iiwa_description/urdf/"
                "iiwa14_polytope_collision.urdf";
                
                // for reading files
                std::ifstream input_file;
                std::string input_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/";
                
                // problem parameters
                int N = 40;
                //double T = 10.0;
                int num_states = 14;
                //int num_inputs = 7;
                
                /* READ FILE AND MAKE TRAJECTORY */
                PiecewisePolynomial<double> readFileAndMakeTrajectory(std::string filename) {
                    // filename
                    std::string traj_filename = input_folder + filename + ".txt";
                    
                    // open output file
                    input_file.open(traj_filename);
                    if (!input_file.is_open()) {
                        std::cerr << "Problem opening file at " << traj_filename << std::endl;
                    }
                    
                    // state metric
                    //std::vector<MatrixXd> state(N);
                    Eigen::MatrixXd knots = Eigen::MatrixXd::Zero(num_states, N + 1);
                    Eigen::MatrixXd breaks = Eigen::VectorXd::Zero(N + 1);
                    
                    // read values (note that breaks and knots are shifted by one column to hold first point)
                    for (int i = 0; i < N; i++) {
                        input_file >> breaks(i + 1);
                        for (int ii = 0; ii < num_states; ii++) {
                            input_file >> knots(ii, i + 1);
                        }
                    }
                    input_file.close();
                    
                    // make first point hold for one second
                    knots.col(0) = knots.col(1);
                    breaks = breaks + Eigen::VectorXd::Ones(N+1);
                    breaks(0) = 0;
                    
                    //std::cout << "Breaks: " << breaks << std::endl;
                    //std::cout << "Knots: " << knots << std::endl;
                    
                    PiecewisePolynomial<double> traj = PiecewisePolynomial<double>::FirstOrderHold(breaks, knots);
                    
                    return traj;
                }
                
                /* MAIN METHOD */
                int DoMain(char* argv1) {
                    DRAKE_DEMAND(FLAGS_simulation_sec > 0);
                    
                    auto tree = std::make_unique<RigidBodyTree<double>>();
                    CreateTreedFromFixedModelAtPose(FindResourceOrThrow(kUrdfPath), tree.get());
                    
                    parsers::urdf::AddModelInstanceFromUrdfFileWithRpyJointToWorld("/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/obstacle.urdf", &(*tree));
                    
                    //PiecewisePolynomial<double> traj = readFileAndMakeTrajectory("admm_al_x_0");
                    //PiecewisePolynomial<double> traj = readFileAndMakeTrajectory("admm_pen_x_0");
                    PiecewisePolynomial<double> traj = readFileAndMakeTrajectory(argv1);
                    
                    drake::lcm::DrakeLcm lcm;
                    SimDiagramBuilder<double> builder;
                    // Adds a plant
                    auto plant = builder.AddPlant(std::move(tree));
                    builder.AddVisualizer(&lcm);
                    
                    // Adds a iiwa controller
                    VectorX<double> iiwa_kp, iiwa_kd, iiwa_ki;
                    SetPositionControlledIiwaGains(&iiwa_kp, &iiwa_ki, &iiwa_kd);
                    
                    auto controller = builder.AddController< systems::controllers::InverseDynamicsController<double>>(RigidBodyTreeConstants::kFirstNonWorldModelInstanceId, plant->get_rigid_body_tree().Clone(), iiwa_kp, iiwa_ki, iiwa_kd, false /* no feedforward acceleration */);
                    
                    // Adds a trajectory source for desired state.
                    systems::DiagramBuilder<double>* diagram_builder =
                    builder.get_mutable_builder();
                    //auto traj_src = diagram_builder->template  AddSystem<systems::TrajectorySource<double>>(traj, 1 /* outputs q + v */);
                    auto traj_src = diagram_builder->template  AddSystem<systems::TrajectorySource<double>>(traj);
                    traj_src->set_name("trajectory_source");
                    
                    std::cout << "traj size: " << traj.rows() << ", " << traj.cols() << std::endl;
                    std::cout << "traj src size: " << traj_src->get_output_port().size() << std::endl;
                    std::cout << "controller size: " << controller->get_input_port_desired_state().size() << std::endl;
                    
                    diagram_builder->Connect(traj_src->get_output_port(), controller->get_input_port_desired_state());
                    
                    std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
                    
                    systems::Simulator<double> simulator(*diagram);
                    simulator.Initialize();
                    simulator.set_target_realtime_rate(1);
                    
                    simulator.StepTo(FLAGS_simulation_sec);
                    
                    return 0;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    if (argc <= 1) {
        std::cout << "Please provide filename extension as an argument to bazel run!" << std::endl;
    }
    
    return drake::examples::kuka_iiwa_arm::DoMain(argv[1]);
}

