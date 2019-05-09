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

namespace drake {
    namespace examples {
        namespace kuka_iiwa_arm {
            namespace {
                
                int DoMain() {
                    
                    const char kUrdfPath[] =
                    "drake/manipulation/models/iiwa_description/urdf/"
                    "iiwa14_polytope_collision.urdf";
                    
                    // define pi, change this later to one of the global constants
                    double pi = 3.14159;
                    
                    // initial and final states
                    Eigen::VectorXd x0(14);
                    Eigen::VectorXd xf(14);
                    
                    x0 << 0, -0.683, 0, 1.77, 0, 0.88, -1.57, 0, 0, 0, 0, 0, 0, 0;
                    xf << 0, 0, 0, -pi/4.0, 0, pi/4.0, pi/2.0, 0, 0, 0, 0, 0, 0, 0;
                    
                    int num_states = 14;
                    int num_inputs = 7;
                    
                    // define time and number of points
                    int N = 40;
                    double T = 4.0;
                    double dt = T/N;
                    
                    // for writing files
                    std::ofstream output_file;
                    std::string output_folder = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/obstacles/";
                    
                    std::unique_ptr<systems::Context<double>> context_ptr;
                    
                    // create tree, plant, builder
                    drake::lcm::DrakeLcm lcm;
                    SimDiagramBuilder<double> builder;
                    
                    auto tree = std::make_unique<RigidBodyTree<double>>();
                    
                    drake::systems::RigidBodyPlant<double>* plant = nullptr;
                    parsers::urdf::AddModelInstanceFromUrdfFileToWorld(FindResourceOrThrow(kUrdfPath), multibody::joints::kFixed, tree.get());
                    plant = builder.AddPlant(std::move(tree));
                    context_ptr = plant->CreateDefaultContext();
                    builder.AddVisualizer(&lcm);
                    
                    const RigidBodyTree<double>& rbtree = plant->get_rigid_body_tree();
                    
                    num_states = plant->get_num_positions() + plant->get_num_velocities();
                    num_inputs = plant->get_num_actuators();
                    
                    // check constants
                    std::cout << "num bodies: " << plant->get_num_bodies() << std::endl;
                    std::cout << "num positions: " << plant->get_num_positions() << std::endl;
                    std::cout << "num actuators: " << plant->get_num_actuators() << std::endl;
                    
                    std::default_random_engine generator;
                    Eigen::VectorXd q = rbtree.getRandomConfiguration(generator);
                    KinematicsCache<double> cache_x0 = rbtree.doKinematics(x0.segment(0, 7), x0.segment(7, 7));
                    KinematicsCache<double> cache_xf = rbtree.doKinematics(xf.segment(0, 7), xf.segment(7, 7));
                    
                    Eigen::Matrix<double, 3, -1> points = Eigen::Matrix<double, 3, -1>::Zero(3, 1);
                    
                    std::cout << "\ntransform points:" << std::endl;
                    std::cout << rbtree.transformPoints(cache_x0, points, 6, 0) << std::endl;
                    std::cout << rbtree.transformPoints(cache_xf, points, 6, 0) << std::endl;
                    std::cout << "\ntransform points Jacobian:" << std::endl;
                    std::cout << rbtree.transformPointsJacobian(cache_x0, points, 6, 0, false) << std::endl;
                    std::cout << rbtree.transformPointsJacobian(cache_xf, points, 6, 0, false) << std::endl;
                    
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

// need to do...
// (done): find how to resolve error with transformPoints
// (done): get points and Jacobian
// decide where to put cylindrical obstacle
// write down constraint in pseudocode
// implement constraint
// write method to calculate end-effector trajectory given a vector, just to see if it looks plausible, and add to printing at end of file
// if code runs, run example
// while code runs, plot!


