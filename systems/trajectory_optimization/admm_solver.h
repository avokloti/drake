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
#include "drake/common/eigen_types.h"

#define DEBUG 1

using namespace std;
using namespace Eigen;

namespace drake {
namespace systems {
namespace trajectory_optimization {
namespace admm_solver {
            
enum constraint_flag {EQUALITY, INEQUALITY};

//typedef void (*function_pointer)(double, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>);
//typedef void (*constraint_function) (double, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>);

typedef std::function<void(double, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>)> single_constraint_function;
    
typedef std::function<void(double, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>)> double_constraint_function;

struct single_constraint_struct {
    single_constraint_function function;
    constraint_flag flag;
    int length;
    std::string constraint_name;
};
    
struct double_constraint_struct {
    double_constraint_function function;
    constraint_flag flag;
    int length;
    std::string constraint_name;
};

    

class AdmmSolver {
    /* --------- fields --------- */
private:
    int N; // number of knot points
    double T; // total time
    double dt; // time step
    
    systems::System<double>* system; // main dynamical system
    std::unique_ptr<systems::Context<double> > context; // context for system
    FixedInputPortValue* input_port_value;
    
    // autodiff versions
    std::unique_ptr<System<AutoDiffXd>> autodiff_system;
    std::unique_ptr<Context<AutoDiffXd>> autodiff_context;
    
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
    //function_pointer dynamics;
    // function handle to a Matlab function that evaluates all constraints
    // function handle to a Matlab function that evaluates dynamics
    int num_constraints;
    std::vector<single_constraint_struct> single_constraints_list;
    std::vector<double_constraint_struct> double_constraints_list;
    //std::vector<constraint_flag> constraint_flag_list;
    
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
    Eigen::MatrixXd solution_x;
    Eigen::MatrixXd solution_u;
    
   ofstream output_file; // for printing debug info
    
    /* --------- functions --------- */
public:
    // constructor:
    AdmmSolver(systems::System<double>* par_system, Eigen::VectorXd par_x0, Eigen::VectorXd par_xf, double par_T, int par_N, int par_max_iter);
    
    // typedef for dynamics function pointer
    //typedef void (*function_pointer)(double, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd>, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>);
    
    // setter functions:
    void setRho1(double rho1);
    void setRho2(double rho2);
    void setRho3(double rho3);
    void setFeasibilityTolerance(double tol);
    void setConstraintTolerance(double tol);
    
    // bounds and constraints
    void setStateUpperBound(Eigen::Ref<Eigen::VectorXd> bound);
    void setStateLowerBound(Eigen::Ref<Eigen::VectorXd> bound);
    void setInputUpperBound(Eigen::Ref<Eigen::VectorXd> bound);
    void setInputLowerBound(Eigen::Ref<Eigen::VectorXd> bound);
    
    // reconstruct
    trajectories::PiecewisePolynomial<double> reconstructStateTrajectory();
    trajectories::PiecewisePolynomial<double> reconstructInputTrajectory();
    Eigen::MatrixXd getSolutionStateTrajectory();
    Eigen::MatrixXd getSolutionInputTrajectory();
    
    void setCostMatrices(Eigen::MatrixXd Q, Eigen::MatrixXd R);
    void addEqualityConstraint();
    void addInequalityConstraint();
    //void addConstraintToAllKnotPoints(constraint_function f, int constraint_size, std::string constraint_name);
    void addInequalityConstraintToAllKnotPoints(single_constraint_function f, int constraint_size, std::string constraint_name);
    void addEqualityConstraintToAllKnotPoints(single_constraint_function f, int constraint_size, std::string constraint_name);
    void addInequalityConstraintToConsecutiveKnotPoints(double_constraint_function f, int constraint_size, std::string constraint_name);
    void addEqualityConstraintToConsecutiveKnotPoints(double_constraint_function f, int constraint_size, std::string constraint_name);
    
    // dynamics for testing
    static void quadDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii);
    
    // solve method:
    void solve(Eigen::Ref<Eigen::VectorXd> y);
    
private:
    double getStateFromY(Eigen::Ref<Eigen::VectorXd> y, int time_index, int index);
    double getInputFromY(Eigen::Ref<Eigen::VectorXd> y, int time_index, int index);
    
    void integratorDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> mid_state, Eigen::Ref<Eigen::VectorXd> mid_input, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii);
    //void quadDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii);
    
    void placeAinM(std::vector<Triplet<double> >* tripletsM, Eigen::Ref<Eigen::MatrixXd> Aii, int ii);
    void placeBinM(std::vector<Triplet<double> >* tripletsM, Eigen::Ref<Eigen::MatrixXd> Bii, int ii);
    void placeinG(std::vector<Triplet<double> >* tripletsGptr, Eigen::Ref<Eigen::MatrixXd> single_dg_x, Eigen::Ref<Eigen::MatrixXd> single_dg_u, int ii, int running_constraint_counter, int cf_length);
    
    void makeCVector(Eigen::Ref<Eigen::VectorXd> c, int ii, Eigen::Ref<Eigen::VectorXd> mid_state, Eigen::Ref<Eigen::VectorXd> mid_input, Eigen::Ref<Eigen::VectorXd> fii, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii, Eigen::Ref<Eigen::VectorXd> y);
    
    Eigen::VectorXd proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> nu, Eigen::Ref<Eigen::MatrixXd> R, double rho1);
    Eigen::VectorXd proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> nu, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c,  Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3);
    
    // upper and lower bounds (private functions)
    void inputLowerBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u);
    void inputUpperBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u);
    void stateLowerBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u);
    void stateUpperBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u);
};
} // namespace admm_solver
} // namespace trajectory_optimization
} // namespace systems
} // namespace drake


