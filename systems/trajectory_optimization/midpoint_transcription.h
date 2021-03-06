#pragma once

#include <memory>
#include <fstream>

#include "drake/common/drake_copyable.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/system.h"
#include "drake/systems/trajectory_optimization/multiple_shooting.h"
#include "drake/multibody/rigid_body_tree.h"

namespace drake {
    namespace systems {
        namespace trajectory_optimization {
            
            /// MidpointTranscription implements the approach to trajectory optimization as
            /// described in
            ///   C. R. Hargraves and S. W. Paris. Direct trajectory optimization using
            ///    nonlinear programming and collocation. J Guidance, 10(4):338-342,
            ///    July-August 1987.
            /// It assumes a first-order hold on the input trajectory and a cubic spline
            /// representation of the state trajectory, and adds dynamic constraints (and
            /// running costs) to the midpoints as well as the knot points in order to
            /// achieve a 3rd order integration accuracy.
            class MidpointTranscription : public MultipleShooting {
            public:
                DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MidpointTranscription)
                
                /// Constructs the %MathematicalProgram% and adds the collocation constraints.
                ///
                /// @param system A dynamical system to be used in the dynamic constraints.
                ///    This system must support System::ToAutoDiffXd.
                ///    Note that this is aliased for the lifetime of this object.
                /// @param context Required to describe any parameters of the system.  The
                ///    values of the state in this context do not have any effect.  This
                ///    context will also be "cloned" by the optimization; changes to the
                ///    context after calling this method will NOT impact the trajectory
                ///    optimization.
                /// @param num_time_samples The number of knot points in the trajectory.
                /// @param minimum_timestep Minimum spacing between sample times.
                /// @param maximum_timestep Maximum spacing between sample times.
                MidpointTranscription(const System<double>* system,
                                  const Context<double>& context, int num_time_samples,
                                  double minimum_timestep, double maximum_timestep);
                
                // NOTE: The fixed timestep constructor, which would avoid adding h as
                // decision variables, has been removed since it complicates the API and code.
                // Unlike other trajectory optimization transcriptions, direct collocation
                // will not be a convex optimization even if the sample times are fixed, so
                // there is little advantage to actually removing the variables.  Setting
                // minimum_timestep == maximum_timestep should be essentially just as good.
                
                ~MidpointTranscription() override {}
                
                trajectories::PiecewisePolynomial<double> ReconstructInputTrajectory()
                const override;
                
                trajectories::PiecewisePolynomial<double> ReconstructStateTrajectory()
                const override;
                
                Eigen::MatrixXd getStateTrajectoryMatrix(int num_states) const;
                
                Eigen::MatrixXd getInputTrajectoryMatrix(int num_inputs) const;
                
                void AddInterpolatedObstacleConstraintToAllPoints(Eigen::Ref<Eigen::VectorXd> obstacle_center_x, Eigen::Ref<Eigen::VectorXd> obstacle_center_y, Eigen::Ref<Eigen::VectorXd> obstacle_radii_x, Eigen::Ref<Eigen::VectorXd> obstacle_radii_y, int num_alpha);
                
                void AddTaskSpaceObstacleConstraintToAllPoints(Eigen::Ref<Eigen::VectorXd> obstacle_center_x, Eigen::Ref<Eigen::VectorXd> obstacle_center_y, Eigen::Ref<Eigen::VectorXd> obstacle_radii_x, Eigen::Ref<Eigen::VectorXd> obstacle_radii_y, RigidBodyTree<double>& rbtree);
                
                void AddPrintingConstraintToAllPoints(std::ofstream &x_stream, std::ofstream &u_stream);
                
            private:
                // Implements a running cost at all timesteps using trapezoidal integration.
                void DoAddRunningCost(const symbolic::Expression& e) override;
                
                // Store system-relevant data for e.g. computing the derivatives during
                // trajectory reconstruction.
                const System<double>* system_{nullptr};
                const std::unique_ptr<Context<double>> context_{nullptr};
                const std::unique_ptr<ContinuousState<double>> continuous_state_{nullptr};
                FixedInputPortValue* input_port_value_{nullptr};
                
                double timestep;
            };
            
            /// Implements the direct collocation constraints for a first-order hold on
            /// the input and a cubic polynomial representation of the state trajectories.
            ///
            /// Note that the MidpointTranscription implementation allocates only ONE of
            /// these constraints, but binds that constraint multiple times (with
            /// different decision variables, along the trajectory).
            
            class MidpointTranscriptionConstraint : public solvers::Constraint {
            public:
                DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MidpointTranscriptionConstraint)
                
            public:
                MidpointTranscriptionConstraint(const System<double>& system,
                                            const Context<double>& context);
                
                ~MidpointTranscriptionConstraint() override = default;
                
                int num_states() const { return num_states_; }
                int num_inputs() const { return num_inputs_; }
                
                
                /// Helper method to add a MidpointTranscriptionConstraint to the @p prog,
                /// ensuring that the order of variables in the binding matches the order
                /// expected by the constraint.
                // Note: The order of arguments is a compromise between GSG and the desire to
                // match the AddConstraint interfaces in MathematicalProgram.
                solvers::Binding<solvers::Constraint> AddMidpointTranscriptionConstraint(std::shared_ptr<MidpointTranscriptionConstraint> constraint,
                                                                                         const Eigen::Ref<const solvers::VectorXDecisionVariable>& timestep,
                                                                                         const Eigen::Ref<const solvers::VectorXDecisionVariable>& state,
                                                                                         const Eigen::Ref<const solvers::VectorXDecisionVariable>& next_state,
                                                                                         const Eigen::Ref<const solvers::VectorXDecisionVariable>& input,
                                                                                         const Eigen::Ref<const solvers::VectorXDecisionVariable>& next_input,
                                                                                         solvers::MathematicalProgram* prog);
                
            protected:
                void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::VectorXd* y) const override;
                
                void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                            AutoDiffVecXd* y) const override;
                
                void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                            VectorX<symbolic::Expression>* y) const override;
                
            private:
                MidpointTranscriptionConstraint(const System<double>& system,
                                            const Context<double>& context, int num_states,
                                            int num_inputs);
                
                void dynamics(const AutoDiffVecXd& state, const AutoDiffVecXd& input,
                              AutoDiffVecXd* xdot) const;
                
                std::unique_ptr<System<AutoDiffXd>> system_;
                std::unique_ptr<Context<AutoDiffXd>> context_;
                FixedInputPortValue* input_port_value_{nullptr};
                std::unique_ptr<ContinuousState<AutoDiffXd>> derivatives_;
                
                const int num_states_{0};
                const int num_inputs_{0};
            };
            
            /// Helper method to add a MidpointTranscriptionConstraint to the @p prog,
            /// ensuring that the order of variables in the binding matches the order
            /// expected by the constraint.
            // Note: The order of arguments is a compromise between GSG and the desire to
            // match the AddConstraint interfaces in MathematicalProgram.
            solvers::Binding<solvers::Constraint> AddMidpointTranscriptionConstraint(
                                                                                 std::shared_ptr<MidpointTranscriptionConstraint> constraint,
                                                                                 const Eigen::Ref<const solvers::VectorXDecisionVariable>& timestep,
                                                                                 const Eigen::Ref<const solvers::VectorXDecisionVariable>& state,
                                                                                 const Eigen::Ref<const solvers::VectorXDecisionVariable>& next_state,
                                                                                 const Eigen::Ref<const solvers::VectorXDecisionVariable>& input,
                                                                                 const Eigen::Ref<const solvers::VectorXDecisionVariable>& next_input,
                                                                                 solvers::MathematicalProgram* prog);
            
            
            
            /// Implements the interpolated object avoidance constraints.
            ///
            /// Note that the MidpointTranscription implementation allocates only ONE of
            /// these constraints, but binds that constraint multiple times (with
            /// different decision variables, along the trajectory). (??? left from previous code)
            
            class InterpolatedObstacleConstraint : public solvers::Constraint {
            public:
                DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(InterpolatedObstacleConstraint)
                
            public:
                InterpolatedObstacleConstraint(int num_states, int num_inputs, Eigen::Ref<Eigen::VectorXd> obstacle_center_x, Eigen::Ref<Eigen::VectorXd> obstacle_center_y, Eigen::Ref<Eigen::VectorXd> obstacle_radii_x, Eigen::Ref<Eigen::VectorXd> obstacle_radii_y, int num_alpha);
                
                ~InterpolatedObstacleConstraint() override = default;
                int num_states() const { return num_states_; }
                int num_inputs() const { return num_inputs_; }
                
            protected:
                void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::VectorXd* y) const override;
                
                void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                            AutoDiffVecXd* y) const override;
                
                void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                            VectorX<symbolic::Expression>* y) const override;
                
            private:
                InterpolatedObstacleConstraint(const System<double>& system,
                                                const Context<double>& context, int num_states,
                                                int num_inputs);
                
                //std::unique_ptr<System<AutoDiffXd>> system_;
                //std::unique_ptr<Context<AutoDiffXd>> context_;
                //FixedInputPortValue* input_port_value_{nullptr};
                //std::unique_ptr<ContinuousState<AutoDiffXd>> derivatives_;
                
                int num_states_;
                int num_inputs_;
                Eigen::VectorXd obstacle_center_x_;
                Eigen::VectorXd obstacle_center_y_;
                Eigen::VectorXd obstacle_radii_x_;
                Eigen::VectorXd obstacle_radii_y_;
                int num_alpha_;
            };
            
            /// Helper method to add a MidpointTranscriptionConstraint to the @p prog,
            /// ensuring that the order of variables in the binding matches the order
            /// expected by the constraint.
            // Note: The order of arguments is a compromise between GSG and the desire to
            // match the AddConstraint interfaces in MathematicalProgram.
            solvers::Binding<solvers::Constraint> AddInterpolatedObstacleConstraint(
                                                                                     std::shared_ptr<InterpolatedObstacleConstraint> constraint,
                                                                                     const Eigen::Ref<const solvers::VectorXDecisionVariable>& timestep,
                                                                                     const Eigen::Ref<const solvers::VectorXDecisionVariable>& state,
                                                                                     const Eigen::Ref<const solvers::VectorXDecisionVariable>& next_state,
                                                                                     const Eigen::Ref<const solvers::VectorXDecisionVariable>& input,
                                                                                     const Eigen::Ref<const solvers::VectorXDecisionVariable>& next_input,
                                                                                     solvers::MathematicalProgram* prog);
            
            /// Implements the task-space object avoidance constraints.
            ///
            /// Note that the MidpointTranscription implementation allocates only ONE of
            /// these constraints, but binds that constraint multiple times (with
            /// different decision variables, along the trajectory). (??? left from previous code)
            
            class TaskSpaceObstacleConstraint : public solvers::Constraint {
            public:
                DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TaskSpaceObstacleConstraint)
                
            public:
                TaskSpaceObstacleConstraint(int num_states, int num_inputs, Eigen::Ref<Eigen::VectorXd> obstacle_center_x, Eigen::Ref<Eigen::VectorXd> obstacle_center_y, Eigen::Ref<Eigen::VectorXd> obstacle_radii_x, Eigen::Ref<Eigen::VectorXd> obstacle_radii_y, RigidBodyTree<double>& rbtree);
                
                ~TaskSpaceObstacleConstraint() override = default;
                int num_states() const { return num_states_; }
                int num_inputs() const { return num_inputs_; }
                
            protected:
                void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::VectorXd* y) const override;
                
                void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                            AutoDiffVecXd* y) const override;
                
                void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                            VectorX<symbolic::Expression>* y) const override;
                
            private:
                TaskSpaceObstacleConstraint(const System<double>& system,
                                               const Context<double>& context, int num_states,
                                               int num_inputs);
                
                //std::unique_ptr<System<AutoDiffXd>> system_;
                //std::unique_ptr<Context<AutoDiffXd>> context_;
                //FixedInputPortValue* input_port_value_{nullptr};
                //std::unique_ptr<ContinuousState<AutoDiffXd>> derivatives_;
                
                int num_states_;
                int num_inputs_;
                int num_obstacles_;
                Eigen::VectorXd obstacle_center_x_;
                Eigen::VectorXd obstacle_center_y_;
                Eigen::VectorXd obstacle_radii_x_;
                Eigen::VectorXd obstacle_radii_y_;
                std::unique_ptr<RigidBodyTree<double>> rbtree_;
            };
            
            /// Helper method to add a MidpointTranscriptionConstraint to the @p prog,
            /// ensuring that the order of variables in the binding matches the order
            /// expected by the constraint.
            // Note: The order of arguments is a compromise between GSG and the desire to
            // match the AddConstraint interfaces in MathematicalProgram.
            solvers::Binding<solvers::Constraint> AddTaskSpaceObstacleConstraint(
                                                                                    std::shared_ptr<InterpolatedObstacleConstraint> constraint,
                                                                                    const Eigen::Ref<const solvers::VectorXDecisionVariable>& timestep,
                                                                                    const Eigen::Ref<const solvers::VectorXDecisionVariable>& state,
                                                                                    const Eigen::Ref<const solvers::VectorXDecisionVariable>& input,
                                                                                    solvers::MathematicalProgram* prog);
            
            
            /// Fake "constraint" just used for printing trajectory.
            class PrintingConstraint : public solvers::Constraint {
                public:
                DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PrintingConstraint)
                
                public:
                PrintingConstraint(std::ofstream &x_stream, std::ofstream &u_stream, int num_states, int num_inputs, int N);
                
                ~PrintingConstraint() override = default;
                int num_states() const { return num_states_; }
                int num_inputs() const { return num_inputs_; }
                int N() const { return N_; }
                
                protected:
                void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::VectorXd* y) const override;
                
                void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                            AutoDiffVecXd* y) const override;
                
                void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                            VectorX<symbolic::Expression>* y) const override;
                
                private:
                //PrintingConstraint(const System<double>& system, const Context<double>& context, int num_states, int num_inputs);
                
                int num_states_;
                int num_inputs_;
                int N_;
                std::ofstream &x_stream_;
                std::ofstream &u_stream_;
            };
            
            /// Helper method to add a MidpointTranscriptionConstraint to the @p prog,
            /// ensuring that the order of variables in the binding matches the order
            /// expected by the constraint.
            // Note: The order of arguments is a compromise between GSG and the desire to
            // match the AddConstraint interfaces in MathematicalProgram.
            solvers::Binding<solvers::Constraint> AddPrintingConstraint(
                                                                        std::shared_ptr<PrintingConstraint> constraint,
                                                                         const Eigen::Ref<const solvers::VectorXDecisionVariable>& state,
                                                                         const Eigen::Ref<const solvers::VectorXDecisionVariable>& input,
                                                                         solvers::MathematicalProgram* prog);
            
            
        }  // namespace trajectory_optimization
    }  // namespace systems
}  // namespace drake
