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
#include "drake/systems/trajectory_optimization/midpoint_transcription.h"

DEFINE_double(simulation_sec, 1, "Number of seconds to simulate.");

using drake::geometry::SceneGraph;
using drake::lcm::DrakeLcm;
using drake::multibody::Body;
using drake::multibody::multibody_plant::MultibodyPlant;
using drake::multibody::MultibodyTree;
using drake::multibody::parsing::AddModelFromSdfFile;
using drake::multibody::UniformGravityFieldElement;

namespace drake {
    namespace examples {
        namespace kuka_iiwa_arm {
            namespace {
                using trajectories::PiecewisePolynomial;
                
                const char kSdfPath[] =
                "drake/manipulation/models/iiwa_description/sdf/"
                "iiwa14_no_collision.sdf";
                
                int DoMain() {
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
                    systems::trajectory_optimization::MidpointTranscription traj_opt(kuka_plant, kuka_plant.CreateDefaultContext(), 10, 0.5, 0.5);
                    
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

