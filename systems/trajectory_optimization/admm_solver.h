#include <stdio.h>
#include <iostream>
#include <chrono>
#include <Eigen/Eigen>

#include "drake/systems/framework/system.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/common/trajectories/piecewise_polynomial.h"

using namespace std;
using namespace Eigen;

namespace drake {
namespace systems {
namespace trajectory_optimization {
            
enum constraint_type {EQUALITY, INEQUALITY};

typedef void (*function_pointer)(double, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>);

class AdmmSolver {
    /* --------- fields --------- */
private:
    int N; // number of knot points
    double T; // total time
    double dt; // time step
    
    System<double>* system; // main dynamical system
    std::unique_ptr<Context<double> > context; // context for system
    FixedInputPortValue* input_port_value;
    
    double initial_rho1; // proximal parameter
    double initial_rho2; // weight on feasibility quadratic cost
    double initial_rho3; // weight on constraint quadratic cost
    
    double rho1_decrease_rate; // decrease by a constant factor if constraints are satisfied
    double rho2_increase_rate; // increases by a constant factor with every iteration
    double rho3_increase_rate; // increases by a constant factor with every iteration
    double rho_max; // maximum value for rho2 and rho3
    double rho1_min;
    
    //?? objective_cost; // store cost in a better way... for now using terms below:
    Eigen::MatrixXd costQ; // x' Q x term
    Eigen::MatrixXd costR; // u' R u term
    
    // function pointers
    function_pointer dynamics;
    // function handle to a Matlab function that evaluates all constraints
    // function handle to a Matlab function that evaluates dynamics
    int num_constraints;
    Eigen::VectorXd x_lower_bound;
    Eigen::VectorXd x_upper_bound;
    Eigen::VectorXd u_lower_bound;
    Eigen::VectorXd u_upper_bound;
    
    int num_states;
    int num_inputs;
    
    Eigen::VectorXd initial_x; // initialization of x trajectory
    Eigen::VectorXd initial_u; // initialization of u trajectory
    
    Eigen::VectorXd x0; // start state
    Eigen::VectorXd xf; // final state
    
    double tol_feasibility; // tolerance on feasibility of trajectory
    double tol_constraints; // tolerance on satisfaction of all other constraints
    double tol_objective; // tolerance on objective minimization (?)
    int max_iter; // maximum number of iterations
    bool solve_flag;
    
    /* --------- functions --------- */
public:
    // constructor:
    AdmmSolver(System<double>* par_system, double* par_x0, double* par_xf, double par_T, int par_N, int par_max_iter);
    
    // typedef for dynamics function pointer
    //typedef void (*function_pointer)(double, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>);
    
    // setter functions:
    void setRho1(double rho1);
    void setRho2(double rho2);
    void setRho3(double rho3);
    void setFeasibilityTolerance(double tol);
    
    // bounds and constraints
    void setStateUpperBound(Eigen::Ref<Eigen::VectorXd> bound);
    void setStateLowerBound(Eigen::Ref<Eigen::VectorXd> bound);
    void setInputUpperBound(Eigen::Ref<Eigen::VectorXd> bound);
    void setInputLowerBound(Eigen::Ref<Eigen::VectorXd> bound);
    
    // reconstruct
    trajectories::PiecewisePolynomial<double> reconstructStateTrajectory();
    trajectories::PiecewisePolynomial<double> reconstructInputTrajectory();
    
    void setCostMatrices(Eigen::MatrixXd Q, Eigen::MatrixXd R);
    void addEqualityConstraint();
    void addInequalityConstraint();
    
    // dynamics for testing
    static void quadDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii);
    
    // solve method:
    void solve(Eigen::Ref<Eigen::VectorXd> y);
    
private:
    double getStateFromY(Eigen::Ref<Eigen::VectorXd> y, int time_index, int index);
    double getInputFromY(Eigen::Ref<Eigen::VectorXd> y, int time_index, int index);
    
    void integratorDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> mid_state, Eigen::Ref<Eigen::VectorXd> mid_input, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii);
    //void quadDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii);
    
    void placeAinM(std::vector<Triplet<double> >* tripletsM, Eigen::Ref<Eigen::MatrixXd> Aii, int ii, double dt);
    void placeBinM(std::vector<Triplet<double> >* tripletsM, Eigen::Ref<Eigen::MatrixXd> Bii, int ii, double dt);
    
    void makeCVector(Eigen::Ref<Eigen::VectorXd> c, int ii, Eigen::Ref<Eigen::VectorXd> mid_state, Eigen::Ref<Eigen::VectorXd> mid_input, Eigen::Ref<Eigen::VectorXd> fii, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii, Eigen::Ref<Eigen::VectorXd> y);
    
    Eigen::VectorXd proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> nu, Eigen::Ref<Eigen::MatrixXd> R, double rho1);
    Eigen::VectorXd proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> nu, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, double rho1, double rho2);
    
    
};
        
        } // namespace trajectory_optimization
    } // namespace systems
} // namespace drake


