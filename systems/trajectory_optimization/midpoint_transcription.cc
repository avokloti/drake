#include "drake/systems/trajectory_optimization/midpoint_transcription.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
    namespace systems {
        namespace trajectory_optimization {
            
            using trajectories::PiecewisePolynomial;
            using solvers::Binding;
            using solvers::Constraint;
            using solvers::MathematicalProgram;
            using solvers::VectorXDecisionVariable;
            
            /* -------------- MidpointTranscriptionConstraint methods -------------- */
            
            MidpointTranscriptionConstraint::MidpointTranscriptionConstraint(const System<double>& system, const Context<double>& context): MidpointTranscriptionConstraint(system, context, context.get_continuous_state().size(),
                                          (context.get_num_input_ports() > 0 ? system.get_input_port(0).size() : 0)) {}
            
            MidpointTranscriptionConstraint::MidpointTranscriptionConstraint(const System<double>& system, const Context<double>& context, int num_states, int num_inputs) : Constraint(num_states, 1 + (2 * num_states) + (2 * num_inputs),
                         Eigen::VectorXd::Zero(num_states),
                         Eigen::VectorXd::Zero(num_states)),
            system_(System<double>::ToAutoDiffXd(system)),
            context_(system_->CreateDefaultContext()),
            // Don't allocate the input port until we're past the point
            // where we might throw.
            derivatives_(system_->AllocateTimeDerivatives()),
            num_states_(num_states),
            num_inputs_(num_inputs) {
                DRAKE_THROW_UNLESS(system_->get_num_input_ports() <= 1);
                DRAKE_THROW_UNLESS(context.has_only_continuous_state());
                
                // TODO(russt): Add support for time-varying dynamics OR check for
                // time-invariance.
                
                context_->SetTimeStateAndParametersFrom(context);
                
                if (context.get_num_input_ports() > 0) {
                    // Allocate the input port and keep an alias around.
                    input_port_value_ = &context_->FixInputPort(
                                                                0, system_->AllocateInputVector(system_->get_input_port(0)));
                }
            }
            
            void MidpointTranscriptionConstraint::dynamics(const AutoDiffVecXd& state,
                                                       const AutoDiffVecXd& input,
                                                       AutoDiffVecXd* xdot) const {
                if (context_->get_num_input_ports() > 0) {
                    input_port_value_->GetMutableVectorData<AutoDiffXd>()->SetFromVector(input);
                }
                context_->get_mutable_continuous_state().SetFromVector(state);
                system_->CalcTimeDerivatives(*context_, derivatives_.get());
                *xdot = derivatives_->CopyToVector();
            }
            
            void MidpointTranscriptionConstraint::DoEval(
                                                     const Eigen::Ref<const Eigen::VectorXd>& x,
                                                     Eigen::VectorXd* y) const {
                AutoDiffVecXd y_t;
                Eval(math::initializeAutoDiff(x), &y_t);
                *y = math::autoDiffToValueMatrix(y_t);
            }
            
            // The format of the input to the eval() function is the
            // tuple { timestep, state 0, state 1, input 0, input 1 },
            // which has a total length of 1 + 2*num_states + 2*num_inputs.
            void MidpointTranscriptionConstraint::DoEval(
                                                     const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
                DRAKE_ASSERT(x.size() == 1 + (2 * num_states_) + (2 * num_inputs_));
                
                // Extract our input variables:
                // h - current time (knot) value
                // x0, x1 state vector at time steps k, k+1
                // u0, u1 input vector at time steps k, k+1
                const AutoDiffXd h = x(0);
                const auto x0 = x.segment(1, num_states_);
                const auto x1 = x.segment(1 + num_states_, num_states_);
                const auto u0 = x.segment(1 + (2 * num_states_), num_inputs_);
                const auto u1 = x.segment(1 + (2 * num_states_) + num_inputs_, num_inputs_);
                
                // TODO(sam.creasey): Use caching (when it arrives) to avoid recomputing
                // the dynamics.  Currently the dynamics evaluated here as {u1,x1} are
                // recomputed in the next constraint as {u0,x0}.
                /*
                AutoDiffVecXd xdot0;
                dynamics(x0, u0, &xdot0);
                const Eigen::MatrixXd dxdot0 = math::autoDiffToGradientMatrix(xdot0);
                
                AutoDiffVecXd xdot1;
                dynamics(x1, u1, &xdot1);
                const Eigen::MatrixXd dxdot1 = math::autoDiffToGradientMatrix(xdot1); */
                
                // midpoint
                AutoDiffVecXd g;
                dynamics(0.5 * (x0 + x1), 0.5 * (u0 + u1), &g);
                *y = x1 - (x0 + h * g);
            }
            
            void MidpointTranscriptionConstraint::DoEval(
                                                     const Eigen::Ref<const VectorX<symbolic::Variable>>&,
                                                     VectorX<symbolic::Expression>*) const {
                throw std::logic_error(
                                       "MidpointTranscriptionConstraint does not support symbolic evaluation.");
            }
            
            Binding<Constraint> AddMidpointTranscriptionConstraint(
                                                               std::shared_ptr<MidpointTranscriptionConstraint> constraint,
                                                               const Eigen::Ref<const VectorXDecisionVariable>& timestep,
                                                               const Eigen::Ref<const VectorXDecisionVariable>& state,
                                                               const Eigen::Ref<const VectorXDecisionVariable>& next_state,
                                                               const Eigen::Ref<const VectorXDecisionVariable>& input,
                                                               const Eigen::Ref<const VectorXDecisionVariable>& next_input,
                                                               MathematicalProgram* prog) {
                DRAKE_DEMAND(timestep.size() == 1);
                DRAKE_DEMAND(state.size() == constraint->num_states());
                DRAKE_DEMAND(next_state.size() == constraint->num_states());
                DRAKE_DEMAND(input.size() == constraint->num_inputs());
                DRAKE_DEMAND(next_input.size() == constraint->num_inputs());
                return prog->AddConstraint(constraint,
                                           {timestep, state, next_state, input, next_input});
            }
            
            /* -------------- MidpointTranscription methods -------------- */
            
            MidpointTranscription::MidpointTranscription(const System<double>* system,
                                                 const Context<double>& context,
                                                 int num_time_samples,
                                                 double minimum_timestep,
                                                 double maximum_timestep)
            : MultipleShooting(system->get_num_total_inputs(),
                               context.get_continuous_state().size(), num_time_samples,
                               minimum_timestep, maximum_timestep),
            system_(system),
            context_(system_->CreateDefaultContext()),
            continuous_state_(system_->AllocateTimeDerivatives()) {
                DRAKE_DEMAND(context.has_only_continuous_state());
                DRAKE_DEMAND(minimum_timestep == maximum_timestep);
                timestep = minimum_timestep;
                
                context_->SetTimeStateAndParametersFrom(context);
                
                if (context_->get_num_input_ports() > 0) {
                    // Allocate the input port and keep an alias around.
                    input_port_value_ = &context_->FixInputPort(
                                                                0, system_->AllocateInputVector(system_->get_input_port(0)));
                }
                
                // Add the dynamic constraints.
                auto constraint = std::make_shared<MidpointTranscriptionConstraint>(
                                                                                *system, context);
                
                DRAKE_ASSERT(static_cast<int>(constraint->num_constraints()) == num_states());
                
                // For N-1 timesteps, add a constraint which depends on the knot
                // value along with the state and input vectors at that knot and the
                // next.
                for (int i = 0; i < N() - 1; i++) {
                    AddConstraint(constraint,
                                  {h_vars().segment<1>(i),
                                      x_vars().segment(i * num_states(), num_states() * 2),
                                      u_vars().segment(i * num_inputs(), num_inputs() * 2)});
                }
            }
            
            void MidpointTranscription::AddInterpolatedObstacleConstraintToAllPoints(Eigen::Ref<Eigen::VectorXd> obstacle_center_x, Eigen::Ref<Eigen::VectorXd> obstacle_center_y, Eigen::Ref<Eigen::VectorXd> obstacle_radii_x, Eigen::Ref<Eigen::VectorXd> obstacle_radii_y, int num_alpha) {
                
                // Add the dynamic constraints.
                auto constraint = std::make_shared<InterpolatedObstacleConstraint>(num_states(), num_inputs(), obstacle_center_x, obstacle_center_y, obstacle_radii_x, obstacle_radii_y, num_alpha);
                
                DRAKE_ASSERT(static_cast<int>(constraint->num_constraints()) == num_alpha * obstacle_center_x.size());
                
                // For N-1 timesteps, add a constraint which depends on the knot
                // value along with the state and input vectors at that knot and the
                // next.
                for (int i = 0; i < N() - 1; i++) {
                    AddConstraint(constraint,
                                  {h_vars().segment<1>(i),
                                      x_vars().segment(i * num_states(), num_states() * 2),
                                      u_vars().segment(i * num_inputs(), num_inputs() * 2)});
                }
            }
            
            void MidpointTranscription::DoAddRunningCost(const symbolic::Expression& g) {
                // Trapezoidal integration:
                //    sum_{i=0...N-2} h_i/2.0 * (g_i + g_{i+1}), or
                // g_0*h_0/2.0 + [sum_{i=1...N-2} g_i*(h_{i-1} + h_i)/2.0] +
                // g_{N-1}*h_{N-2}/2.0.
                
                //AddCost(SubstitutePlaceholderVariables(g * h_vars()(0) / 2, 0));
                //for (int i = 1; i <= N() - 2; i++) {
                //    AddCost(SubstitutePlaceholderVariables(
                //                                           g * (h_vars()(i - 1) + h_vars()(i)) / 2, i));
                //}
                //AddCost(SubstitutePlaceholderVariables(g * h_vars()(N() - 2) / 2, N() - 1));
                for (int i = 0; i < N() - 1; i++) {
                    AddCost(SubstitutePlaceholderVariables(g * timestep, i));
                }
                // assumes minimum_timestep = maximum_timestep
            }
            
            PiecewisePolynomial<double>
            MidpointTranscription::ReconstructInputTrajectory()
            const {
                DRAKE_DEMAND(context_->get_num_input_ports() > 0);
                Eigen::VectorXd times = GetSampleTimes();
                std::vector<double> times_vec(N());
                std::vector<Eigen::MatrixXd> inputs(N());
                
                for (int i = 0; i < N(); i++) {
                    times_vec[i] = times(i);
                    inputs[i] = GetSolution(input(i));
                }
                return PiecewisePolynomial<double>::FirstOrderHold(times_vec, inputs);
            }
            
            PiecewisePolynomial<double>
            MidpointTranscription::ReconstructStateTrajectory()
            const {
                Eigen::VectorXd times = GetSampleTimes();
                std::vector<double> times_vec(N());
                std::vector<Eigen::MatrixXd> states(N());
                std::vector<Eigen::MatrixXd> derivatives(N());
                
                for (int i = 0; i < N(); i++) {
                    times_vec[i] = times(i);
                    states[i] = GetSolution(state(i));
                    if (context_->get_num_input_ports() > 0) {
                        input_port_value_->GetMutableVectorData<double>()->SetFromVector(
                                                                                         GetSolution(input(i)));
                    }
                    context_->get_mutable_continuous_state().SetFromVector(states[i]);
                    system_->CalcTimeDerivatives(*context_, continuous_state_.get());
                    derivatives[i] = continuous_state_->CopyToVector();
                }
                return PiecewisePolynomial<double>::Cubic(times_vec, states, derivatives);
            }
            
            Eigen::MatrixXd MidpointTranscription::getStateTrajectoryMatrix(int num_states) const {
                Eigen::VectorXd times = GetSampleTimes();
                Eigen::MatrixXd states(num_states, N());
                
                for (int i = 0; i < N(); i++) {
                    states.col(i) = GetSolution(state(i));
                    //if (context_->get_num_input_ports() > 0) {
                    //    input_port_value_->GetMutableVectorData<double>()->SetFromVector(GetSolution(input(i)));
                    //}
                    //context_->get_mutable_continuous_state().SetFromVector(states.col(i));
                    //system_->CalcTimeDerivatives(*context_, continuous_state_.get());
                    //states.block(num_states/2, i, num_states/2, 1) = continuous_state_->CopyToVector();
                }
                return states;
            }
            
            Eigen::MatrixXd MidpointTranscription::getInputTrajectoryMatrix(int num_inputs) const {
                Eigen::VectorXd times = GetSampleTimes();
                Eigen::MatrixXd inputs(num_inputs, N());
                
                for (int i = 0; i < N(); i++) {
                    inputs.col(i) = GetSolution(input(i));
                }
                return inputs;
            }
            
            /* INTERPOLATED OBSTACLE AVOIDANCE CONSTRAINT METHODS */
            InterpolatedObstacleConstraint::InterpolatedObstacleConstraint(int num_states, int num_inputs, Eigen::Ref<Eigen::VectorXd> obstacle_center_x, Eigen::Ref<Eigen::VectorXd> obstacle_center_y, Eigen::Ref<Eigen::VectorXd> obstacle_radii_x, Eigen::Ref<Eigen::VectorXd> obstacle_radii_y, int num_alpha): Constraint(num_alpha * obstacle_radii_x.size(), 1 + (2 * num_states) + (2 * num_inputs), -100000 * Eigen::VectorXd::Ones(obstacle_radii_x.size() * num_alpha), Eigen::VectorXd::Zero(obstacle_radii_x.size() * num_alpha)) {
                
                std::cout << "num states: " << num_states << std::endl;
                std::cout << "num inputs: " << num_inputs << std::endl;
                std::cout << "num alpha: " << num_alpha << std::endl;
                std::cout << "constraint size: " << num_alpha * obstacle_radii_x.size() << std::endl;
                
                num_states_ = num_states;
                num_inputs_ = num_inputs;
                obstacle_center_x_ = obstacle_center_x;
                obstacle_center_y_ = obstacle_center_y;
                obstacle_radii_x_ = obstacle_radii_x;
                obstacle_radii_y_ = obstacle_radii_y;
                num_alpha_ = num_alpha;
            }
            
            void InterpolatedObstacleConstraint::DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const {
                AutoDiffVecXd y_t;
                Eval(math::initializeAutoDiff(x), &y_t);
                *y = math::autoDiffToValueMatrix(y_t);
            }
            
            // The format of the input to the eval() function is the
            // tuple { timestep, state 0, state 1, input 0, input 1 },
            // which has a total length of 1 + 2*num_states + 2*num_inputs.
            void InterpolatedObstacleConstraint::DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
                DRAKE_ASSERT(x.size() == 1 + (2 * num_states_) + (2 * num_inputs_));
                
                const AutoDiffXd h = x(0);
                const auto x1 = x.segment(1, num_states_);
                const auto x2 = x.segment(1 + num_states_, num_states_);
                //const auto u1 = x.segment(1 + (2 * num_states_), num_inputs_);
                //const auto u2 = x.segment(1 + (2 * num_states_) + num_inputs_, num_inputs_);
                
                std::vector<double> alpha;
                for (int i = 0; i < num_alpha_; i++) {
                    alpha.push_back(static_cast<double>(i)/static_cast<double>(num_alpha_));
                }
                
                // uncomment later
                for (int i = 0; i < obstacle_radii_x_.size(); i++) {
                    for (int ii = 0; ii < num_alpha_; ii++) {
                        // entries of d
                        int index = i * num_alpha_ + ii;
                        (*y)(index) = 1 -
                        ((1 - alpha[ii]) * x1[0] + alpha[ii] * x2[0] - obstacle_center_x_[i]) *
                        ((1 - alpha[ii]) * x1[0] + alpha[ii] * x2[0] - obstacle_center_x_[i])/(obstacle_radii_x_[i] * obstacle_radii_x_[i]) -
                        ((1 - alpha[ii]) * x1[1] + alpha[ii] * x2[1] - obstacle_center_y_[i]) *
                        ((1 - alpha[ii]) * x1[1] + alpha[ii] * x2[1] - obstacle_center_y_[i])/(obstacle_radii_y_[i] * obstacle_radii_y_[i]);
                    }
                }
            }
            
            void InterpolatedObstacleConstraint::DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
                                                         VectorX<symbolic::Expression>*) const {
                throw std::logic_error("MidpointTranscriptionConstraint does not support symbolic evaluation.");
            }
            
            Binding<Constraint> AddInterpolatedObstacleConstraint(std::shared_ptr<InterpolatedObstacleConstraint> constraint,
                                                                   const Eigen::Ref<const VectorXDecisionVariable>& timestep,
                                                                   const Eigen::Ref<const VectorXDecisionVariable>& state,
                                                                   const Eigen::Ref<const VectorXDecisionVariable>& next_state,
                                                                   const Eigen::Ref<const VectorXDecisionVariable>& input,
                                                                   const Eigen::Ref<const VectorXDecisionVariable>& next_input,
                                                                   MathematicalProgram* prog) {
                DRAKE_DEMAND(timestep.size() == 1);
                DRAKE_DEMAND(state.size() == constraint->num_states());
                DRAKE_DEMAND(next_state.size() == constraint->num_states());
                DRAKE_DEMAND(input.size() == constraint->num_inputs());
                DRAKE_DEMAND(next_input.size() == constraint->num_inputs());
                return prog->AddConstraint(constraint,
                                           {timestep, state, next_state, input, next_input});
            }
            
        }  // namespace trajectory_optimization
    }  // namespace systems
}  // namespace drake
