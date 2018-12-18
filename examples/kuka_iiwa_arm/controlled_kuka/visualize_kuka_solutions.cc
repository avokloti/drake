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

DEFINE_double(simulation_sec, 1, "Number of seconds to simulate.");

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
                std::string input_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/basic/";
                
                // problem parameters
                int N = 10;
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
                    Eigen::MatrixXd knots = Eigen::MatrixXd::Zero(num_states, N);
                    Eigen::MatrixXd breaks = Eigen::VectorXd::Zero(N);
                    
                    // read values
                    for (int i = 0; i < N; i++) {
                        input_file >> breaks(i);
                        for (int ii = 0; ii < num_states; ii++) {
                            input_file >> knots(ii, i);
                        }
                    }
                    input_file.close();
                    
                    PiecewisePolynomial<double> traj = PiecewisePolynomial<double>::FirstOrderHold(breaks, knots);
                    
                    return traj;
                }
                
                /* MAIN METHOD */
                int DoMain() {
                    DRAKE_DEMAND(FLAGS_simulation_sec > 0);
                    
                    auto tree = std::make_unique<RigidBodyTree<double>>();
                    CreateTreedFromFixedModelAtPose(FindResourceOrThrow(kUrdfPath), tree.get());
                    
                    PiecewisePolynomial<double> traj = readFileAndMakeTrajectory("ipopt_x_0");
                    
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
                    auto traj_src = diagram_builder->template  AddSystem<systems::TrajectorySource<double>>(traj, 1 /* outputs q + v */);
                    traj_src->set_name("trajectory_source");
                    
                    diagram_builder->Connect(traj_src->get_output_port(), controller->get_input_port_desired_state());
                    
                    std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
                    
                    systems::Simulator<double> simulator(*diagram);
                    simulator.Initialize();
                    simulator.set_target_realtime_rate(1.0);
                    
                    simulator.StepTo(FLAGS_simulation_sec);
                    
                    return 0;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    return drake::examples::kuka_iiwa_arm::DoMain();
}


