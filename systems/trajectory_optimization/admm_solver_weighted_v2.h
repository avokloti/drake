/*
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Eigen>
#include <functional>

#include "drake/systems/framework/system.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h" */
#include "drake/systems/trajectory_optimization/admm_solver_base.h"

using namespace std;
using namespace Eigen;

namespace drake {
    namespace systems {
        namespace trajectory_optimization {
            namespace admm_solver {
                
                class AdmmSolverWeightedV2 : public AdmmSolverBase {
                    
                    /* --------- functions --------- */
                public:
                    // constructor:
                    AdmmSolverWeightedV2(systems::System<double>* par_system, Eigen::VectorXd par_x0, Eigen::VectorXd par_xf, double par_T, int par_N, int par_max_iter);
                    
                    AdmmSolverWeightedV2(systems::System<double>* par_system);
                    
                    ~AdmmSolverWeightedV2() {};
                    
                    // solve method:
                    std::string solve(Eigen::Ref<Eigen::VectorXd> y);
                
                protected:
                    //Eigen::VectorXd proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> y, Eigen::Ref<Eigen::VectorXd> lambda1, Eigen::Ref<Eigen::MatrixXd> R, Eigen::Ref<Eigen::VectorXd> b, double rho1);
                    Eigen::VectorXd proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> lambda1, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3);
                };
            } //namespace accel_admm_solver
        } // namespace trajectory_optimization
    } // namespace systems
} // namespace drake


