#pragma once

#include <memory>

#include "drake/common/symbolic.h"
#include "drake/examples/complete_robobee/gen/complete_robobee_input.h"
#include "drake/examples/complete_robobee/gen/complete_robobee_params.h"
#include "drake/examples/complete_robobee/gen/complete_robobee_state.h"
#include "drake/geometry/scene_graph.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
    namespace examples {
        namespace complete_robobee {
            
            /// @tparam T The vector element type, which must be a valid Eigen scalar.
            ///
            /// Instantiated templates for the following kinds of T's are provided:
            ///
            /// - double
            /// - AutoDiffXd
            /// - symbolic::Expression
            template <typename T>
            class CompleteRobobeePlant final : public systems::LeafSystem<T> {
            public:
                //DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CompleteRobobeePlant);
                
                CompleteRobobeePlant();
                
                /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
                template <typename U>
                explicit CompleteRobobeePlant(const CompleteRobobeePlant<U>&);
                
                ~CompleteRobobeePlant() override;
                
                /// Returns the input port to the externally applied force.
                const systems::InputPort<T>& get_input_port() const;
                
                /// Returns the port to output state.
                const systems::OutputPort<T>& get_state_output_port() const;
                
                geometry::SourceId source_id() const { return source_id_; }
                geometry::FrameId frame_id() const { return frame_id_; }
                
                static const CompleteRobobeeState<T>& get_state(const systems::ContinuousState<T>& cstate) {
                    return dynamic_cast<const CompleteRobobeeState<T>&>(cstate.get_vector());
                }
                
                static const CompleteRobobeeState<T>& get_state(const systems::Context<T>& context) {
                    return get_state(context.get_continuous_state());
                }
                
                static CompleteRobobeeState<T>& get_mutable_state(systems::ContinuousState<T>* cstate) {
                    return dynamic_cast<CompleteRobobeeState<T>&>(cstate->get_mutable_vector());
                }
                
                static CompleteRobobeeState<T>& get_mutable_state(systems::Context<T>* context) {
                    return get_mutable_state(&context->get_mutable_continuous_state());
                }
                
                const CompleteRobobeeParams<T>& get_parameters(const systems::Context<T>& context) const {
                    return this->template GetNumericParameter<CompleteRobobeeParams>(context, 0);
                }
                
                const T& get_w(const systems::Context<T>& context) const {
                    return this->EvalVectorInput(context, 0)->GetAtIndex(0);
                }
                
                const T& get_V_avg(const systems::Context<T>& context) const {
                    return this->EvalVectorInput(context, 0)->GetAtIndex(1);
                }
                
                const T& get_V_dif(const systems::Context<T>& context) const {
                    return this->EvalVectorInput(context, 0)->GetAtIndex(2);
                }
                
                const T& get_V_off(const systems::Context<T>& context) const {
                    return this->EvalVectorInput(context, 0)->GetAtIndex(3);
                }
                
                std::vector<double> GetInputBounds() const;
                
                CompleteRobobeeParams<T>& get_mutable_parameters(systems::Context<T>* context) const {
                    return this->template GetMutableNumericParameter<CompleteRobobeeParams>(context, 0);
                }
                
                
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
            
        }  // namespace complete_robobee
    }  // namespace examples
}  // namespace drake
