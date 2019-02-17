#include "drake/systems/trajectory_optimization/admm_solver_base.h"

using namespace std;
using namespace Eigen;

namespace drake {
    namespace systems {
        namespace trajectory_optimization {
            namespace admm_solver {
                
                class ALISolver : public AdmmSolverBase {

                public:
                    // constructor:
                    ALISolver(systems::System<double>* par_system, Eigen::VectorXd par_x0, Eigen::VectorXd par_xf, double par_T, int par_N, int par_max_iter);
                    
                    ALISolver(systems::System<double>* par_system);
                    
                    ~ALISolver() {};
                    
                    // solve method:
                    std::string solve(Eigen::Ref<Eigen::VectorXd> y);
                    
                protected:
                    Eigen::VectorXd proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> lambda1, Eigen::Ref<Eigen::VectorXd> lambda2, Eigen::Ref<Eigen::VectorXd> lambda3, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3);
                };
            } //namespace admm_solver
        } // namespace trajectory_optimization
    } // namespace systems
} // namespace drake


