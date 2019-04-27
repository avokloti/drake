#include "drake/examples/complete_robobee/complete_robobee_plant.h"

#include <cmath>
#include <vector>

#include "drake/common/drake_throw.h"
#include "drake/common/default_scalars.h"
#include "drake/geometry/geometry_frame.h"
#include "drake/geometry/geometry_instance.h"
#include "drake/math/rotation_matrix.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
    namespace examples {
        namespace complete_robobee {
            
            using Eigen::Isometry3d;
            using Eigen::Translation3d;
            using Eigen::Vector4d;
            using geometry::Box;
            using geometry::Cylinder;
            using geometry::GeometryFrame;
            using geometry::GeometryInstance;
            using geometry::Sphere;
            using geometry::VisualMaterial;
            using std::make_unique;
            
            template <typename T>
            CompleteRobobeePlant<T>::CompleteRobobeePlant() : systems::LeafSystem<T>(systems::SystemTypeTag<complete_robobee::CompleteRobobeePlant>{}) {
                //this->DeclareVectorInputPort(RobobeeInput<T>());
                this->DeclareVectorOutputPort(systems::BasicVector<T>(12), &CompleteRobobeePlant::CopyStateOut);
                this->DeclareContinuousState(CompleteRobobeeState<T>(), 6, 6, 0);
                this->DeclareInputPort(systems::kVectorValued, 4);
                this->DeclareNumericParameter(CompleteRobobeeParams<T>());
            }
            
            template <typename T>
            template <typename U>
            CompleteRobobeePlant<T>::CompleteRobobeePlant(const CompleteRobobeePlant<U>& p) : CompleteRobobeePlant() {}
            
            template <typename T>
            CompleteRobobeePlant<T>::~CompleteRobobeePlant() {}
            
            template <typename T>
            const systems::InputPort<T>& CompleteRobobeePlant<T>::get_input_port() const {
                return systems::System<T>::get_input_port(0);
            }
            
            template <typename T>
            const systems::OutputPort<T>& CompleteRobobeePlant<T>::get_state_output_port() const {
                return systems::System<T>::get_output_port(0); // not sure what indexing should be here
            }
            
            template <typename T>
            void CompleteRobobeePlant<T>::CopyStateOut(const systems::Context<T> &context, systems::BasicVector<T> *output) const {
                output->set_value(context.get_continuous_state_vector().CopyToVector());
            }
            
            // Compute the actual physics.
            template <typename T>
            void CompleteRobobeePlant<T>::DoCalcTimeDerivatives(const systems::Context<T>& context, systems::ContinuousState<T>* derivatives) const {
                
                // get robobee state and parameters
                const CompleteRobobeeState<T>& state = get_state(context);
                const CompleteRobobeeParams<T>& params = get_parameters(context);
                
                // get specific inputs values
                const T& w = get_w(context);
                const T& V_avg = get_V_avg(context);
                const T& V_dif = get_V_dif(context);
                const T& V_off = get_V_off(context);
                
                // precompute soem values
                const T& rho_B_Cl = params.rho() * params.B() * params.Cl();
                
                const T& G0 = params.A() / params.k_eq();
                
                const T& w_Gw_squared = (params.A() * params.A() * w * w) /
                (params.m_eq() * params.m_eq() * w * w * w * w + (params.b_eq() * params.b_eq() - 2 * params.m_eq() * params.k_eq()) * w * w + params.k_eq() * params.k_eq());
                
                // calculate input values from voltage and w values
                const T& F_t = 0.5 * rho_B_Cl * w_Gw_squared * (V_avg * V_avg + V_dif * V_dif);
                const T& tau_alpha = params.r_cp() * rho_B_Cl * w_Gw_squared * (V_avg * V_dif);
                const T& tau_beta = params.r_cp() * V_off * G0 * F_t;
                const T& tau_gamma = 0; // ASSUMES KAPPA = 0.5
                
                // construct (and fill in) this derivative_vector
                CompleteRobobeeState<T>& derivative_vector = get_mutable_state(derivatives);
                
                // set velocities
                derivative_vector.set_x(state.x_dot());
                derivative_vector.set_y(state.y_dot());
                derivative_vector.set_z(state.z_dot());
                
                // set angular velocities
                derivative_vector.set_alpha(state.alpha_dot());
                derivative_vector.set_beta(state.beta_dot());
                derivative_vector.set_gamma(state.gamma_dot());
                
                // set accelerations
                derivative_vector.set_x_dot(F_t * sin(state.beta()) / params.m());
                derivative_vector.set_y_dot(-F_t * sin(state.alpha()) * cos(state.beta()) / params.m());
                derivative_vector.set_z_dot(F_t * cos(state.alpha()) * cos(state.beta()) / params.m() - params.g());
                
                // set angular accelerations
                derivative_vector.set_alpha_dot(tau_alpha / params.I_alpha());
                derivative_vector.set_beta_dot(tau_beta / params.I_beta());
                derivative_vector.set_gamma_dot(tau_gamma / params.I_gamma());
            }
        }
    }
}

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(class ::drake::examples::complete_robobee::CompleteRobobeePlant)
