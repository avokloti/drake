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
                
                // want to do:
                // make obstacleConstraints return g and dg
                // run same test: calculate (g(q) - g(q + dq[i]))/dq[i] for each i, compare against dg[i]
                
                void obstacleConstraints(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> y0, Eigen::Ref<Eigen::MatrixXd> dy, const RigidBodyTree<double>* rbtree) {
                    
                    int num_obstacles_ = 1;
                    
                    Eigen::VectorXd obstacle_center_x_(1);
                    Eigen::VectorXd obstacle_center_y_(1);
                    Eigen::VectorXd obstacle_radii_x_(1);
                    Eigen::VectorXd obstacle_radii_y_(1);
                    
                    obstacle_center_x_ << -0.2;
                    obstacle_center_y_ << 0.05;
                    obstacle_radii_x_ << 0.05;
                    obstacle_radii_y_ << 0.05;
                    
                    // prepare for kinematics calculations
                    Eigen::VectorXd q = x.segment(0, 7);
                    Eigen::VectorXd v = x.segment(7, 7);
                    auto cache = rbtree->doKinematics(q, v);
                    Eigen::Matrix<double, 3, -1> points = Eigen::Matrix<double, 3, -1>::Zero(3, 1);
                    
                    // calculate end-effector position and Jacobian
                    auto ee = rbtree->transformPoints(cache, points, 10, 0);
                    Eigen::MatrixXd ee_jacobian = rbtree->transformPointsJacobian(cache, points, 10, 0, false);
                    
                    // place correctly in constraint matrix
                    for (int i = 0; i < num_obstacles_; i++) {
                        y0(i) = 1 - (obstacle_center_x_[i] - ee(0)) * (obstacle_center_x_[i] - ee(0))/(obstacle_radii_x_[i] * obstacle_radii_x_[i]) - (obstacle_center_y_[i] - ee(1)) * (obstacle_center_y_[i] - ee(1))/(obstacle_radii_y_[i] * obstacle_radii_y_[i]);
                        
                        // entries of dy
                        double aa = (ee(0) - obstacle_center_x_[i])/(obstacle_radii_x_[i] * obstacle_radii_x_[i]);
                        double bb = (ee(1) - obstacle_center_y_[i])/(obstacle_radii_y_[i] * obstacle_radii_y_[i]);
                        Eigen::VectorXd gradient = -2 * (aa * ee_jacobian.row(0) + bb * ee_jacobian.row(1));
                        //std::cout << "\n" << gradient.transpose() << "\n" << std::endl;
                        dy.block(i, 0, 1, 7) = gradient.transpose();
                    }
                }
                
                int DoMain() {
                    
                    const char kUrdfPath[] =
                    "drake/manipulation/models/iiwa_description/urdf/"
                    "iiwa14_polytope_collision.urdf";
                    
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
                    
                    const RigidBodyTree<double> &rbtree = plant->get_rigid_body_tree();
                    
                    int num_states = plant->get_num_positions() + plant->get_num_velocities();
                    int num_inputs = plant->get_num_actuators();
                    
                    // check constants
                    std::cout << "num bodies: " << plant->get_num_bodies() << std::endl;
                    std::cout << "num frames: " << rbtree.get_num_frames() << std::endl;
                    std::cout << "num positions: " << plant->get_num_positions() << std::endl;
                    std::cout << "num actuators: " << plant->get_num_actuators() << std::endl;
                    
                    for (int i=0; i < int(rbtree.get_num_bodies()); i++) {
                        std::cout << "Name of frame " << i << ": " << rbtree.getBodyOrFrameName(i) << std::endl;
                    }
                    
                    
                    // ---------------------------------------------------------------------
                    // numerically test full gradient for obstacleConstraints
                    
                    // prepare constraint vector and gradient matrix
                    int num_obstacles_ = 1;
                    
                    Eigen::VectorXd y0 = Eigen::VectorXd::Zero(num_obstacles_);
                    Eigen::MatrixXd dy0 = Eigen::MatrixXd::Zero(num_obstacles_, num_states);
                    
                    Eigen::VectorXd yf = Eigen::VectorXd::Zero(num_obstacles_);
                    Eigen::MatrixXd dyf = Eigen::MatrixXd::Zero(num_obstacles_, num_states);
                    
                    std::default_random_engine generator;
                    
                    for (int trial = 0; trial < 10; trial++) {
                        
                        Eigen::VectorXd x0 = rbtree.getRandomConfiguration(generator);
                        Eigen::VectorXd xf = x0;
                        
                        // calculate constraints for y0
                        obstacleConstraints(x0, y0, dy0, &rbtree);
                        
                        Eigen::MatrixXd numerical_diff = Eigen::MatrixXd::Zero(num_obstacles_, num_states/2);
                        
                        for (int i = 0; i < num_states/2; i++) {
                            // make small step in ith direction
                            xf[i] = xf[i] + 1e-6;
                            
                            // calculate constraints for yf
                            obstacleConstraints(xf, yf, dyf, &rbtree);
                            
                            // fill in numerical derivative
                            numerical_diff.col(i) = (yf - y0)/(1e-6);
                            
                            // step back in ith direction
                            xf[i] = xf[i] - 1e-6;
                        }
                        
                        std::cout << "Numerical diff:\n" << numerical_diff << std::endl;
                        
                        std::cout << "My method and transformPoints:\n" << dy0 << std::endl;
                        
                        std::cout << "\n\n" << std::endl;
                    }
                    
                    
                    
                    
                    /* ---------------------------------------------------------------------
                    // numerically test whether derivatives of transformPoints are accurate
                    std::default_random_engine generator;
                    
                    for (int trial = 0; trial < 10; trial++) {
                        Eigen::VectorXd x0 = rbtree.getRandomConfiguration(generator);
                        Eigen::VectorXd xf = x0;
                        
                        Eigen::Matrix<double, 3, -1> points = Eigen::Matrix<double, 3, -1>::Zero(3, 1);
                        
                        // try testing very basic numerical differentiation?
                        KinematicsCache<double> cache_x0 = rbtree.doKinematics(x0.segment(0, 7), x0.segment(7, 7));
                        auto x0_vector = rbtree.transformPoints(cache_x0, points, 10, 0);
                        Eigen::MatrixXd numerical_diff = Eigen::MatrixXd::Zero(3, num_states/2);
                        
                        for (int i = 0; i < num_states/2; i++) {
                            xf[i] = xf[i] + 0.0000001;
                            KinematicsCache<double> cache_xf = rbtree.doKinematics(xf.segment(0, 7), xf.segment(7, 7));
                            auto xf_vector = rbtree.transformPoints(cache_xf, points, 10, 0);
                            numerical_diff.col(i) = (xf_vector - x0_vector)/0.0000001;
                            xf[i] = xf[i] - 0.0000001;
                        }
                        
                        std::cout << "Numerical diff - transformPointJacobian results:\n" << numerical_diff - rbtree.transformPointsJacobian(cache_x0, points, 10, 0, false) << std::endl;
                        
                        std::cout << "\n\n" << std::endl;
                    } */
                    
                    
                    
                    /* ---------------------------------------------------------------------
                    // test transformPoints across different values
                    std::cout << "\ntransform points at x0:" << std::endl;
                    std::cout << rbtree.transformPoints(cache_x0, points, 10, 0) << std::endl;
                    std::cout << "\ntransform points at xf:" << std::endl;
                    std::cout << rbtree.transformPoints(cache_xf, points, 10, 0) << std::endl;
                    std::cout << "\ntransform points at x0, backwards:" << std::endl;
                    std::cout << rbtree.transformPoints(cache_x0, points, 0, 10) << std::endl;
                    std::cout << "\ntransform points at xf, backwards:" << std::endl;
                    std::cout << rbtree.transformPoints(cache_xf, points, 0, 10) << std::endl;
                    std::cout << "\ntransform points Jacobian at x0:" << std::endl;
                    std::cout << rbtree.transformPointsJacobian(cache_x0, points, 10, 0, false) << std::endl;
                    std::cout << "\ntransform points Jacobian at xf:" << std::endl;
                    std::cout << rbtree.transformPointsJacobian(cache_xf, points, 10, 0, false) << std::endl;
                    std::cout << "\ntransform points Jacobian at x0, backwards?:" << std::endl;
                    std::cout << rbtree.transformPointsJacobian(cache_x0, points, 0, 10, false) << std::endl;
                    std::cout << "\ntransform points Jacobian at xf, backwards?:" << std::endl;
                    std::cout << rbtree.transformPointsJacobian(cache_xf, points, 0, 10, false) << std::endl;
                    */
                    
                    
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


