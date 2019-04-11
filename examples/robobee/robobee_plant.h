#pragma once

#include <memory>

#include "drake/common/symbolic.h"
#include "drake/examples/robobee/gen/robobee_input.h"
#include "drake/examples/robobee/gen/robobee_params.h"
#include "drake/examples/robobee/gen/robobee_state.h"
#include "drake/geometry/scene_graph.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
    namespace examples {
        namespace robobee {
            
            /// @tparam T The vector element type, which must be a valid Eigen scalar.
            ///
            /// Instantiated templates for the following kinds of T's are provided:
            ///
            /// - double
            /// - AutoDiffXd
            /// - symbolic::Expression
            template <typename T>
            class RobobeePlant final : public systems::LeafSystem<T> {
            public:
                DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RobobeePlant);
                
                RobobeePlant();
                
                /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
                template <typename U>
                explicit RobobeePlant(const RobobeePlant<U>&);
                
                ~RobobeePlant() override;
                
                /// Returns the input port to the externally applied force.
                const systems::InputPort<T>& get_input_port() const;
                
                /// Returns the port to output state.
                const systems::OutputPort<T>& get_state_output_port() const;
                
                geometry::SourceId source_id() const { return source_id_; }
                geometry::FrameId frame_id() const { return frame_id_; }
                
                static const RobobeeState<T>& get_state(const systems::ContinuousState<T>& cstate) {
                    return dynamic_cast<const RobobeeState<T>&>(cstate.get_vector());
                }
                
                static const RobobeeState<T>& get_state(const systems::Context<T>& context) {
                    return get_state(context.get_continuous_state());
                }
                
                static RobobeeState<T>& get_mutable_state(systems::ContinuousState<T>* cstate) {
                    return dynamic_cast<RobobeeState<T>&>(cstate->get_mutable_vector());
                }
                
                static RobobeeState<T>& get_mutable_state(systems::Context<T>* context) {
                    return get_mutable_state(&context->get_mutable_continuous_state());
                }
                
                const RobobeeParams<T>& get_parameters(const systems::Context<T>& context) const {
                    return this->template GetNumericParameter<RobobeeParams>(context, 0);
                }
                
                const T& get_F_t(const systems::Context<T>& context) const {
                    return this->EvalVectorInput(context, 0)->GetAtIndex(0);
                }
                
                const T& get_tau_alpha(const systems::Context<T>& context) const {
                    return this->EvalVectorInput(context, 0)->GetAtIndex(1);
                }
                
                const T& get_tau_beta(const systems::Context<T>& context) const {
                    return this->EvalVectorInput(context, 0)->GetAtIndex(2);
                }
                
                const T& get_tau_gamma(const systems::Context<T>& context) const {
                    return this->EvalVectorInput(context, 0)->GetAtIndex(3);
                }
                
                RobobeeParams<T>& get_mutable_parameters(systems::Context<T>* context) const {
                    return this->template GetMutableNumericParameter<RobobeeParams>(context, 0);
                }
                
                std::vector<T> SetInputFromVoltage(const systems::Context<T>& context, const T& V_avg, const T& V_diff, const T& V_off, const T& w);
                
            private:
                void CopyStateOut(const systems::Context<T>& context,
                                  systems::BasicVector<T>* output) const;
                
                void DoCalcTimeDerivatives(const systems::Context<T>& context,
                                           systems::ContinuousState<T>* derivatives) const override;
                
                
                // Port handles.
                int state_port_{-1};
                int geometry_pose_port_{-1};
                
                geometry::SourceId source_id_{};
                geometry::FrameId frame_id_{};
            };
            
        }  // namespace robobee
    }  // namespace examples
}  // namespace drake
