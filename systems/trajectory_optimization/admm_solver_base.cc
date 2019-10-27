// Version with obstacle constraints(and sparse M and G)!

// run as: g++ -I /usr/local/include/eigen3/ -O2 solveADMMconstraints.cpp

#include "drake/systems/trajectory_optimization/admm_solver_base.h"

typedef Array<bool,Dynamic,1> ArrayXb;

using namespace std::chrono;
using namespace std::placeholders;

namespace drake {
    namespace systems {
        namespace trajectory_optimization {
            namespace admm_solver {
                
                /* ---------------------------------------------- SOLVER INITIALIZATION ---------------------------------------------- */
                
                AdmmSolverBase::AdmmSolverBase(systems::System<double>* par_system) {
                    // must also set: x0, xf, T, N
                    
                    // set system, context, input ports
                    system = par_system;
                    context = system->CreateDefaultContext();
                    input_port_value = &context->FixInputPort(0, system->AllocateInputVector(system->get_input_port(0)));
                    
                    // create an autodiff version of the system and context
                    autodiff_system = drake::systems::System<double>::ToAutoDiffXd(*system);
                    autodiff_context = autodiff_system->CreateDefaultContext();
                    autodiff_context->SetTimeStateAndParametersFrom(*context);
                    
                    // set num_states and num_inputs
                    num_states = context->get_num_total_states();
                    num_inputs = system->get_input_port(0).size();
                    
                    rho1_decrease_rate = 1.0;// decrease by a constant factor if constraints are satisfied
                    rho2_increase_rate = 1.02; // increases by a constant factor with every iteration
                    rho3_increase_rate = 1.05; // increases by a constant factor with every iteration
                    rho_max = 1e9; // maximum value for rho2 and rho3
                    
                    x0 = Eigen::VectorXd::Zero(num_states);
                    xf = Eigen::VectorXd::Zero(num_states);
                    
                    num_constraints = 0;
                    
                    T = 0;
                    N = 0;
                    dt = 0;
                    max_iter = 4000;
                    
                    solve_flag = false;
                    initial_state_flag = false;
                    final_state_flag = false;
                }
                
                AdmmSolverBase::AdmmSolverBase(systems::System<double>* par_system, Eigen::VectorXd par_x0, Eigen::VectorXd par_xf, double par_T, int par_N, int par_max_iter) {
                    
                    system = par_system;
                    context = system->CreateDefaultContext();
                    input_port_value = &context->FixInputPort(0, system->AllocateInputVector(system->get_input_port(0)));
                    
                    // create an autodiff version of the system and context
                    autodiff_system = drake::systems::System<double>::ToAutoDiffXd(*system);
                    autodiff_context = autodiff_system->CreateDefaultContext();
                    autodiff_context->SetTimeStateAndParametersFrom(*context);
                    
                    rho1_decrease_rate = 1.0; // decrease by a constant factor if constraints are satisfied
                    rho2_increase_rate = 1.02; // increases by a constant factor with every iteration
                    rho3_increase_rate = 1.05; // increases by a constant factor with every iteration
                    rho_max = 1e9; // maximum value for rho2 and rho3
                    
                    num_states = context->get_num_total_states();
                    num_inputs = system->get_input_port(0).size();
                    
                    x0 = Eigen::VectorXd(par_x0);
                    xf = Eigen::VectorXd(par_xf);
                    
                    costQ = Eigen::MatrixXd::Zero(num_states, num_states);
                    costQf = Eigen::MatrixXd::Zero(num_states, num_states);
                    costR = Eigen::MatrixXd::Zero(num_inputs, num_inputs);
                    costq = Eigen::VectorXd::Zero((num_states + num_inputs) * par_N);
                    costqf = Eigen::VectorXd::Zero((num_states + num_inputs) * par_N);
                    costr = Eigen::VectorXd::Zero((num_states + num_inputs) * par_N);
                    
                    num_constraints = 0;
                    
                    T = par_T;
                    N = par_N;
                    dt = T/N;
                    max_iter = par_max_iter;
                    
                    solve_flag = false;
                    initial_state_flag = true;
                    final_state_flag = true;
                }
                
                void AdmmSolverBase::addQuadraticRunningCostOnState(Eigen::Ref<Eigen::MatrixXd> Q, Eigen::Ref<Eigen::VectorXd> q) {
                    costQ = Q;
                    costq = q;
                }
                
                void AdmmSolverBase::addQuadraticFinalCostOnState(Eigen::Ref<Eigen::MatrixXd> Qf, Eigen::Ref<Eigen::VectorXd> qf) {
                    costQf = Qf;
                    costqf = qf;
                }
                
                void AdmmSolverBase::addQuadraticRunningCostOnInput(Eigen::Ref<Eigen::MatrixXd> R, Eigen::Ref<Eigen::VectorXd> r) {
                    costR = R;
                    costr = r;
                }
                
                void AdmmSolverBase::setRho1(double rho) {
                    initial_rho1 = rho;
                }
                
                void AdmmSolverBase::setRho2(double rho) {
                    initial_rho2 = rho;
                }
                
                void AdmmSolverBase::setRho3(double rho) {
                    initial_rho3 = rho;
                }
                
                void AdmmSolverBase::setFeasibilityTolerance(double tol) {
                    tol_feasibility = tol;
                }
                
                void AdmmSolverBase::setConstraintTolerance(double tol) {
                    tol_constraints = tol;
                }
                
                void AdmmSolverBase::setObjectiveTolerance(double tol) {
                    tol_objective = tol;
                }
                
                void AdmmSolverBase::setMaxIterations(int iters) {
                    max_iter = iters;
                }
                
                void AdmmSolverBase::setKnotPoints(int n) {
                    costQ = Eigen::MatrixXd::Zero(num_states, num_states);
                    costQf = Eigen::MatrixXd::Zero(num_states, num_states);
                    costR = Eigen::MatrixXd::Identity(num_inputs, num_inputs);
                    costq = Eigen::VectorXd::Zero((num_states + num_inputs) * n);
                    costqf = Eigen::VectorXd::Zero((num_states + num_inputs) * n);
                    costr = Eigen::VectorXd::Zero((num_states + num_inputs) * n);
                    N = n;
                    dt = T/N;
                }
                
                void AdmmSolverBase::setOutputFile(std::string filename) {
                    output_file = filename;
                }
                void AdmmSolverBase::setTrajFile(std::string filename) {
                    traj_file = filename;
                }
                
                void AdmmSolverBase::setTotalTime(double time) {
                    T = time;
                    dt = T/N;
                }
                
                void AdmmSolverBase::setStartAndEndState(Eigen::Ref<Eigen::VectorXd> start_state, Eigen::Ref<Eigen::VectorXd> end_state) {
                    x0 = start_state;
                    xf = end_state;
                    initial_state_flag = true;
                    final_state_flag = true;
                }
                
                void AdmmSolverBase::setStateUpperBound(Eigen::Ref<Eigen::VectorXd> bound) {
                    if (bound.size() != num_states) {
                        cerr << "State upper bound must have length equal to the number of states.\n";
                    }
                    x_upper_bound = bound;
                    
                    // make constraint (uses lambda to bind "this" object to the member function)
                    single_constraint_function f = [this](double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u){stateUpperBoundConstraint(t, x, u, g, dg_x, dg_u);};
                    
                    struct single_constraint_struct cf = {f, INEQUALITY, num_states, "stateUpperBound"};
                    
                    // add constraint to list
                    single_constraints_list.push_back(cf);
                    
                    // increment overall constraint counter
                    num_constraints = num_constraints + num_states;
                }
                
                void AdmmSolverBase::setStateLowerBound(Eigen::Ref<Eigen::VectorXd> bound) {
                    if (bound.size() != num_states) {
                        cerr << "State lower bound must have length equal to the number of states.\n";
                    }
                    x_lower_bound = bound;
                    
                    // make constraint
                    single_constraint_function f = [this](double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u){stateLowerBoundConstraint(t, x, u, g, dg_x, dg_u);};
                    
                    struct single_constraint_struct cf = {f, INEQUALITY, num_states, "stateLowerBound"};
                    
                    // add constraint to list
                    single_constraints_list.push_back(cf);
                    
                    // increment overall constraint counter
                    num_constraints = num_constraints + num_states;
                }
                
                void AdmmSolverBase::setInputUpperBound(Eigen::Ref<Eigen::VectorXd> bound) {
                    if (bound.size() != num_inputs) {
                        cerr << "Input upper bound must have length equal to the number of inputs.\n";
                    }
                    u_upper_bound = bound;
                    
                    // make constraint
                    single_constraint_function f = [this](double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u){inputUpperBoundConstraint(t, x, u, g, dg_x, dg_u);};
                    
                    struct single_constraint_struct cf = {f, INEQUALITY, num_inputs, "inputUpperBound"};
                    
                    // add constraint to list
                    single_constraints_list.push_back(cf);
                    
                    // increment overall constraint counter
                    num_constraints = num_constraints + num_inputs;
                }
                
                void AdmmSolverBase::setInputLowerBound(Eigen::Ref<Eigen::VectorXd> bound) {
                    if (bound.size() != num_inputs) {
                        cerr << "Input lower bound must have length equal to the number of inputs.\n";
                    }
                    u_lower_bound = bound;
                    
                    // make constraint
                    single_constraint_function f = [this](double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u){inputLowerBoundConstraint(t, x, u, g, dg_x, dg_u);};
                    
                    struct single_constraint_struct cf = {f, INEQUALITY, num_inputs, "inputLowerBound"};
                    
                    // add constraint to list
                    single_constraints_list.push_back(cf);
                    
                    // increment overall constraint counter
                    num_constraints = num_constraints + num_inputs;
                }
                
                Eigen::MatrixXd AdmmSolverBase::getSolutionStateTrajectory() {
                    return solution_x;
                }
                
                Eigen::MatrixXd AdmmSolverBase::getSolutionInputTrajectory() {
                    return solution_u;
                }
                
                Eigen::MatrixXd AdmmSolverBase::getSolutionVector() {
                    return solution_y;
                }
                
                int AdmmSolverBase::getNumLatestIterations() {
                    return num_latest_iterations;
                }
                
                trajectories::PiecewisePolynomial<double> AdmmSolverBase::reconstructStateTrajectory() {
                    // check if problem has been solved
                    if (!solve_flag) {
                        cerr << "Cannot reconstruct trajectory -- problem has not been solved successfully.\n";
                    }
                    
                    std::vector<double> times_vec(N);
                    std::vector<Eigen::MatrixXd> states(N);
                    std::vector<Eigen::MatrixXd> derivatives(N);
                    std::unique_ptr<ContinuousState<double> > continuous_state(system->AllocateTimeDerivatives());
                    
                    for (int i = 0; i < N; i++) {
                        times_vec[i] = dt * i;
                        states[i] = solution_x.block(0, i, num_states, 1);
                        if (context->get_num_input_ports() > 0) {
                            input_port_value->GetMutableVectorData<double>()->SetFromVector(solution_u.block(0, i, num_inputs, 1));
                        }
                        context->get_mutable_continuous_state().SetFromVector(states[i]);
                        system->CalcTimeDerivatives(*context, continuous_state.get());
                        derivatives[i] = continuous_state->CopyToVector();
                    }
                    return trajectories::PiecewisePolynomial<double>::Cubic(times_vec, states, derivatives);
                }
                
                
                trajectories::PiecewisePolynomial<double> AdmmSolverBase::reconstructInputTrajectory() {
                    // check if problem has been solved
                    if (!solve_flag) {
                        cerr << "Cannot reconstruct trajectory -- problem has not been solved successfully.\n";
                    }
                    DRAKE_DEMAND(context->get_num_input_ports() > 0);
                    std::vector<double> times_vec(N);
                    std::vector<Eigen::MatrixXd> inputs(N);
                    
                    for (int i = 0; i < N; i++) {
                        times_vec[i] = dt * i;
                        inputs[i] = solution_u.block(0, i, num_inputs, 1);
                    }
                    return trajectories::PiecewisePolynomial<double>::FirstOrderHold(times_vec, inputs);
                }
                
                void AdmmSolverBase::addInequalityConstraintToAllKnotPoints(single_constraint_function f, int constraint_size, std::string constraint_name) {
                    // make individual constraint structure
                    struct single_constraint_struct cf = {f, INEQUALITY, constraint_size, constraint_name};
                    
                    // put onto overall constraint list
                    single_constraints_list.push_back(cf);
                    num_constraints = num_constraints + constraint_size;
                }
                
                void AdmmSolverBase::addEqualityConstraintToAllKnotPoints(single_constraint_function f, int constraint_size, std::string constraint_name) {
                    
                    // make individual constraint structure
                    struct single_constraint_struct cf = {f, EQUALITY, constraint_size, constraint_name};
                    
                    // put onto overall constraint list
                    single_constraints_list.push_back(cf);
                    num_constraints = num_constraints + constraint_size;
                }
                
                void AdmmSolverBase::addInequalityConstraintToConsecutiveKnotPoints(double_constraint_function f, int constraint_size, std::string constraint_name) {
                    // make individual constraint structure
                    struct double_constraint_struct cf = {f, INEQUALITY, constraint_size, constraint_name};
                    
                    // put onto overall constraint list
                    double_constraints_list.push_back(cf);
                    num_constraints = num_constraints + constraint_size;
                }
                
                void AdmmSolverBase::addEqualityConstraintToConsecutiveKnotPoints(double_constraint_function f, int constraint_size, std::string constraint_name) {
                    // make individual constraint structure
                    struct double_constraint_struct cf = {f, EQUALITY, constraint_size, constraint_name};
                    
                    // put onto overall constraint list
                    double_constraints_list.push_back(cf);
                    num_constraints = num_constraints + constraint_size;
                }
                
                /* ---------------------------------------------- NO SOLVE METHOD ---------------------------------------------- */
                
                
                /* ---------------------------------------------- FUNCTIONS ---------------------------------------------- */
                
                
                // FIRST PROXIMAL UPDATE
                Eigen::VectorXd AdmmSolverBase::proximalUpdateObjective(Eigen::Ref<Eigen::VectorXd> y, Eigen::Ref<Eigen::VectorXd> lambda1, Eigen::Ref<Eigen::MatrixXd> R, Eigen::Ref<Eigen::VectorXd> b, double rho1) {
                    Eigen::MatrixXd temp = 2 * R + rho1 * Eigen::MatrixXd::Identity(N * (num_states + num_inputs), N * (num_states + num_inputs));
                    return temp.llt().solve(rho1 * y - lambda1 - b);
                }
                
                
                // SECOND PROXIMAL UPDATE
                Eigen::VectorXd AdmmSolverBase::proximalUpdateConstraints(Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> lambda1, Eigen::Ref<Eigen::SparseMatrix<double> > M, Eigen::Ref<Eigen::VectorXd> c, Eigen::Ref<Eigen::SparseMatrix<double> > G, Eigen::Ref<Eigen::VectorXd> h, double rho1, double rho2, double rho3) {
                    Eigen::MatrixXd Mt = M.transpose();
                    Eigen::MatrixXd Gt = G.transpose();
                    
                    Eigen::MatrixXd temp = rho1 * Eigen::MatrixXd::Identity(M.cols(), M.cols()) + rho2 * Mt * M + rho3 * Gt * G;
                    return temp.llt().solve(lambda1 + rho1 * x + rho2 * Mt * c + rho3 * Gt * h);
                }
                
                double AdmmSolverBase::getStateFromY(Eigen::Ref<Eigen::VectorXd> y, int time_index, int index) {
                    return (y[time_index * num_states + index]);
                }
                
                double AdmmSolverBase::getInputFromY(Eigen::Ref<Eigen::VectorXd> y, int time_index, int index) {
                    return (y[N * num_states + time_index * num_inputs + index]);
                }
                
                
                // DOUBLE INTEGRATOR DYNAMICS
                void AdmmSolverBase::integratorDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> state, Eigen::Ref<Eigen::VectorXd> input, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii) {
                    f[0] = state[1];
                    f[1] = input[0];
                    
                    Aii(0, 0) = 0; Aii(0, 1) = 1;
                    Aii(1, 0) = 0; Aii(1, 1) = 0;
                    
                    Bii(0, 0) = 0; Bii(1, 0) = 1;
                }
                
                /*
                 // QUAD DYNAMICS
                 void AdmmSolverBase::quadDynamics(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> f, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii) {
                 
                 double quad_L = 0.25; // length of rotor arm
                 double quad_mass = 0.486; // mass of quadrotor
                 double quad_I = 0.00383; // moment of inertia
                 double g = 9.81;
                 
                 f[0] = x[3];
                 f[1] = x[4];
                 f[2] = x[5];
                 f[3] = -sin(x[2])/quad_mass * (u[0] + u[1]);
                 f[4] = -g + cos(x[2])/quad_mass * (u[0] + u[1]);
                 f[5] = quad_L/quad_I * (-u[0] + u[1]);
                 
                 Aii(0, 3) = 1; Aii(1, 4) = 1; Aii(2, 5) = 1;
                 Aii(3, 2) = -cos(x[2])/quad_mass * (u[0] + u[1]);
                 Aii(4, 2) = -sin(x[2])/quad_mass * (u[0] + u[1]);
                 
                 Bii(3, 0) = -sin(x[2])/quad_mass;
                 Bii(3, 1) = -sin(x[2])/quad_mass;
                 Bii(4, 0) = cos(x[2])/quad_mass;
                 Bii(4, 1) = cos(x[2])/quad_mass;
                 Bii(5, 0) = -quad_L/quad_I;
                 Bii(5, 1) = quad_L/quad_I;
                 } */
                
                /*
                 // ALL CONSTRAINTS? (FOR TESTING)
                 void AdmmSolverBase::allConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg) {
                 
                 int num_states = 6; int num_inputs = 2;
                 int N = (x.size() / num_states);
                 
                 // useful
                 int nn = num_inputs + num_states;
                 int num_constraints = 3 * 6 + 2 * (6 + 2);
                 
                 // define state and input constraint values
                 Eigen::VectorXd x_upper_bound(6); x_upper_bound << 6.0, 7.0, 10000, 10000, 10000, 10000;
                 Eigen::VectorXd x_lower_bound(6); x_lower_bound << -1.0, 0.0, -10000, -10000, -10000, -10000;
                 Eigen::VectorXd u_upper_bound(2); u_upper_bound << 10000, 10000;
                 Eigen::VectorXd u_lower_bound(2); u_lower_bound << -10000, -10000;
                 
                 // define obstacles
                 double num_obstacles = 6;
                 struct obstacle obs[6] = {{{1.5, 0.5}, 0.5}, {{1.5, 2.5}, 0.5}, {{1.5, 4.5}, 0.5}, {{3.5, 1.5}, 0.5}, {{3.5, 3.5}, 0.5}, {{3.5, 5.5}, 0.5}};
                 
                 // call state and input bound constraints
                 stateInputConstraints(time_index, x, u, x_upper_bound, x_lower_bound, u_upper_bound, u_lower_bound, num_states, num_inputs, g.segment(0, 2 * nn), dg.block(0, 0, 2 * nn, nn));
                 
                 // call obstacle constraints
                 obstacleConstraints(time_index, x, u, obs, num_obstacles, N, num_states, num_inputs, g.segment(2 * nn, num_obstacles * 3), dg.block(2 * nn, 0, num_obstacles * 3, nn));
                 }
                 */
                
                // STATE INPUT CONSTRAINTS
                void AdmmSolverBase::stateUpperBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                    g = x - x_upper_bound;
                    dg_x = Eigen::MatrixXd::Identity(num_states, num_states);
                    dg_u = Eigen::MatrixXd::Zero(num_states, num_inputs);
                }
                
                void AdmmSolverBase::stateLowerBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                    g = x_lower_bound - x;
                    dg_x = -1 * Eigen::MatrixXd::Identity(num_states, num_states);
                    dg_u = Eigen::MatrixXd::Zero(num_states, num_inputs);
                }
                
                void AdmmSolverBase::inputUpperBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                    g = u - u_upper_bound;
                    dg_x = Eigen::MatrixXd::Zero(num_inputs, num_states);
                    dg_u = Eigen::MatrixXd::Identity(num_inputs, num_inputs);
                }
                
                void AdmmSolverBase::inputLowerBoundConstraint(double t, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg_x, Eigen::Ref<Eigen::MatrixXd> dg_u) {
                    g = u_lower_bound - u;
                    dg_x = Eigen::MatrixXd::Zero(num_inputs, num_states);
                    dg_u = -1 * Eigen::MatrixXd::Identity(num_inputs, num_inputs);
                }
                /*
                 void AdmmSolverBase::stateInputConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, Eigen::Ref<Eigen::VectorXd> x_upper_bound, Eigen::Ref<Eigen::VectorXd> x_lower_bound, Eigen::Ref<Eigen::VectorXd> u_upper_bound, Eigen::Ref<Eigen::VectorXd> u_lower_bound, int num_states, int num_inputs, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg) {
                 
                 int nn = num_states + num_inputs;
                 
                 // entries of g
                 g.segment(0, num_states) = x_lower_bound - x;
                 g.segment(num_states, num_inputs) = u_lower_bound - u;
                 g.segment(num_inputs + num_states, num_states) = x - x_upper_bound;
                 g.segment(num_inputs + 2 * num_states, num_inputs) =  u - u_upper_bound;
                 
                 // entries of dg
                 dg.block(0, 0, nn, nn) = -1 * Eigen::MatrixXd::Identity(nn, nn);
                 dg.block(nn, 0, nn, nn) = Eigen::MatrixXd::Identity(nn, nn);
                 } */
                
                /*
                 // OBSTACLE CONSTRAINTS
                 void AdmmSolverBase::obstacleConstraints(double time_index, Eigen::Ref<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> u, obstacle* obs, int num_constraints, int N, int num_states, int num_inputs, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> dg) {
                 
                 double quad_L = 0.25;
                 
                 double right_side_x, right_side_y;
                 double left_side_x, left_side_y;
                 double rad_squared;
                 
                 for (int i = 0; i < num_constraints; i++) {
                 // pre-compute
                 rad_squared = obs[i].radius * obs[i].radius;
                 right_side_x = x[0] + quad_L * cos(x[2]) - obs[i].center[0];
                 right_side_y = x[1] + quad_L * sin(x[2]) - obs[i].center[1];
                 left_side_x = x[0] - quad_L * cos(x[2]) - obs[i].center[0];
                 left_side_y = x[1] - quad_L * sin(x[2]) - obs[i].center[1];
                 
                 // entries of d
                 g(3 * i) = rad_squared - (obs[i].center[0] - x[0]) * (obs[i].center[0] - x[0]) - (obs[i].center[1] - x[1]) * (obs[i].center[1] - x[1]);
                 g(3 * i + 1) = rad_squared - right_side_x * right_side_x - right_side_y * right_side_y;
                 g(3 * i + 2) = rad_squared - left_side_x * left_side_x - left_side_y * left_side_y;
                 
                 // entries of dd
                 dg(3 * i, 0) = -2 * (x[0] - obs[i].center[0]);
                 dg(3 * i, 1) = -2 * (x[1] - obs[i].center[1]);
                 
                 dg(3 * i + 1, 0) = -2 * right_side_x;
                 dg(3 * i + 1, 1) = -2 * right_side_y;
                 dg(3 * i + 1, 2) = -2 * right_side_x * (-quad_L * sin(x[3])) - 2 * right_side_y * (quad_L * cos(x[3]));
                 
                 dg(3 * i + 2, 0) = -2 * left_side_x;
                 dg(3 * i + 2, 1) = -2 * left_side_y;
                 dg(3 * i + 2, 2) = -2 * left_side_x * (-quad_L * sin(x[3])) - 2 * left_side_y * (quad_L * cos(x[3]));
                 }
                 } */
                
                
                // PLACE A MATRIX INTO M
                void AdmmSolverBase::placeAinM(std::vector<Triplet<double> >* tripletsMptr, Eigen::Ref<Eigen::MatrixXd> Aii, int ii) {
                    // place values of A in M
                    for (int row = 0; row < num_states; row++) {
                        for (int col = 0; col < num_states; col++) {
                            if (row == col){
                                tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, num_states * ii + col, -0.5 * dt * Aii(row, col) - 1));
                                tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, num_states * (ii+1) + col, -0.5 * dt * Aii(row, col) + 1));
                            } else {
                                tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, num_states * ii + col, -0.5 * dt * Aii(row, col)));
                                tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, num_states * (ii+1) + col, -0.5 * dt * Aii(row, col)));
                            }
                        }
                    }
                }
                
                
                // PLACE B MATRIX INTO M
                void AdmmSolverBase::placeBinM(std::vector<Triplet<double> >* tripletsMptr, Eigen::Ref<Eigen::MatrixXd> Bii, int ii) {
                    // should copy entries of Bii into right place in M
                    for (int row = 0; row < num_states; row++) {
                        for (int col = 0; col < num_inputs; col++) {
                            tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, N * num_states + num_inputs * ii + col, -0.5 * dt * Bii(row, col)));
                            tripletsMptr->push_back(Triplet<double>(num_states * (ii+1) + row, N * num_states + num_inputs * (ii+1) + col, -0.5 * dt * Bii(row, col)));
                        }
                    }
                }
                
                
                // CONSTRUCT C VECTOR
                void AdmmSolverBase::makeCVector(Eigen::Ref<Eigen::VectorXd> c, int ii, Eigen::Ref<Eigen::VectorXd> mid_state, Eigen::Ref<Eigen::VectorXd> mid_input, Eigen::Ref<Eigen::VectorXd> fii, Eigen::Ref<Eigen::MatrixXd> Aii, Eigen::Ref<Eigen::MatrixXd> Bii, Eigen::Ref<Eigen::VectorXd> y) {
                    
                    Eigen::VectorXd curr_x = y.segment(ii * num_states, num_states);
                    Eigen::VectorXd curr_u = y.segment(N * num_states + ii * num_inputs, num_inputs);
                    Eigen::VectorXd next_x = y.segment((ii+1) * num_states, num_states);
                    Eigen::VectorXd next_u = y.segment(N * num_states + (ii+1) * num_inputs, num_inputs);
                    
                    c.segment((ii+1) * num_states, num_states) = dt * (fii - 0.5 * Aii * (curr_x + next_x) - 0.5 * Bii * (curr_u + next_u));
                }
                
                
                // PLACE IN G
                /*
                 void AdmmSolverBase::placeinG(int ii, std::vector<Triplet<double> >* tripletsGptr, Eigen::Ref<Eigen::MatrixXd> point_dg_x, Eigen::Ref<Eigen::MatrixXd> point_dg_u) {
                 
                 // place values of point dg's into G
                 for (int row = 0; row < num_constraints; row++) {
                 for (int col = 0; col < num_states; col++) {
                 tripletsGptr->push_back(Triplet<double>(ii * num_constraints + row, ii * num_states + col, point_dg_x(row, col)));
                 }
                 for (int col = 0; col < num_inputs; col++) {
                 tripletsGptr->push_back(Triplet<double>(ii * num_constraints + row, N * num_states + ii * num_inputs + col, point_dg_u(row, col)));
                 }
                 }
                 } */
                
                void AdmmSolverBase::placeinG(std::vector<Triplet<double> >* tripletsGptr, Eigen::Ref<Eigen::MatrixXd> single_dg_x, Eigen::Ref<Eigen::MatrixXd> single_dg_u, int ii, int running_constraint_counter, int cf_length) {
                    
                    for (int row = 0; row < cf_length; row++) {
                        for (int col = 0; col < num_states; col++) {
                            if (single_dg_x(row, col) != 0) {
                                tripletsGptr->push_back(Triplet<double>(ii * num_constraints + running_constraint_counter + row,
                                                                        ii * num_states + col,
                                                                        single_dg_x(row, col)));
                            }
                        }
                    }
                    for (int row = 0; row < cf_length; row++) {
                        for (int col = 0; col < num_inputs; col++) {
                            if (single_dg_u(row, col) != 0) {
                                tripletsGptr->push_back(Triplet<double>(ii * num_constraints + running_constraint_counter + row,
                                                                        N * num_states + ii * num_inputs + col,
                                                                        single_dg_u(row, col)));
                            }
                        }
                    }
                }
                
                //int main(int argc, char* argv[]) {
                    //gflags::ParseCommandLineFlags(&argc, &argv, true);
                //    return 0;
                //}
            } // namespace admm_solver
        } // namespace trajectory_optimization
    } // namespace systems
} // namespace drake


