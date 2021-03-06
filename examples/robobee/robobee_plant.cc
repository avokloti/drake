#include "drake/examples/robobee/robobee_plant.h"

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
        namespace robobee {
            
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
            RobobeePlant<T>::RobobeePlant() : systems::LeafSystem<T>(systems::SystemTypeTag<robobee::RobobeePlant>{}) {
                //this->DeclareVectorInputPort(RobobeeInput<T>());
                this->DeclareVectorOutputPort(systems::BasicVector<T>(12), &RobobeePlant::CopyStateOut);
                this->DeclareContinuousState(RobobeeState<T>(), 6, 6, 0);
                this->DeclareInputPort(systems::kVectorValued, 4);
                this->DeclareNumericParameter(RobobeeParams<T>());
            }
            
            template <typename T>
            template <typename U>
            RobobeePlant<T>::RobobeePlant(const RobobeePlant<U>& p) : RobobeePlant() {}
            
            template <typename T>
            RobobeePlant<T>::~RobobeePlant() {}
            
            template <typename T>
            const systems::InputPort<T>& RobobeePlant<T>::get_input_port() const {
                return systems::System<T>::get_input_port(0);
            }
            
            template <typename T>
            const systems::OutputPort<T>& RobobeePlant<T>::get_state_output_port() const {
                return systems::System<T>::get_output_port(0); // not sure what indexing should be here
            }
            
            template <typename T>
            void RobobeePlant<T>::CopyStateOut(const systems::Context<T> &context, systems::BasicVector<T> *output) const {
                output->set_value(context.get_continuous_state_vector().CopyToVector());
            }
            
            // Compute the actual physics.
            template <typename T>
            void RobobeePlant<T>::DoCalcTimeDerivatives(const systems::Context<T>& context, systems::ContinuousState<T>* derivatives) const {
                
                // get robobee state and parameters
                const RobobeeState<T>& state = get_state(context);
                const RobobeeParams<T>& params = get_parameters(context);
                
                // get specific inputs values
                std::vector<double> scale_factor = {1e4, 1e10, 1e10, 1e10};
                const T& F_t = 1.0/scale_factor.at(0) * get_F_t(context);
                const T& tau_alpha = 1.0/scale_factor.at(1) * get_tau_alpha(context);
                const T& tau_beta = 1.0/scale_factor.at(2) * get_tau_beta(context);
                const T& tau_gamma = 1.0/scale_factor.at(3) * get_tau_gamma(context);
                
                
                // construct (and fill in) this derivative_vector
                RobobeeState<T>& derivative_vector = get_mutable_state(derivatives);
                
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
            
            template <typename T>
            std::vector<T> RobobeePlant<T>::SetInputFromVoltage(const systems::Context<T>& context, const T& V_avg, const T& V_diff, const T& V_off, const T& w) {
                
                const RobobeeParams<T>& params = get_parameters(context);
                
                // precompute
                const T& rho_B_Cl = params.rho() * params.B() * params.Cl();
                
                const T& G0 = params.A() / params.k_eq();
                
                const T& w_Gw_squared = (params.A() * params.A() * w * w) /
                (params.m_eq() * params.m_eq() * w * w * w * w + (params.b_eq() * params.b_eq() - 2 * params.m_eq() * params.k_eq()) * w * w + params.k_eq() * params.k_eq());
                
                // calculate input values from voltage and w values
                const T& F_t = 0.5 * rho_B_Cl * w_Gw_squared * (V_avg * V_avg + V_diff * V_diff);
                
                const T& tau_alpha = params.r_cp() * rho_B_Cl * w_Gw_squared * (V_avg * V_diff);
                
                const T& tau_beta = params.r_cp() * V_off * G0 * F_t;
                
                const T& tau_gamma = 0; // ASSUMES KAPPA = 0.5
                
                // return above values
                std::vector<T> input = {F_t, tau_alpha, tau_beta, tau_gamma};
                return input;
            }
            
            template <typename T>
            std::vector<T> RobobeePlant<T>::GetInputBounds(const systems::Context<T>& context) {
                
                const RobobeeParams<T>& params = get_parameters(context);
                
                // scale
                std::vector<double> scale_factor = {1e4, 1e10, 1e10, 1e10};
                
                // constant
                const T& rho_B_Cl = params.rho() * params.B() * params.Cl();
                
                // constant
                const T& G0 = params.A() / params.k_eq();
                
                // minimum when w is small
                double w_min = 100 * 2 * 3.14159;
                const T& w_Gw_squared_min = (params.A() * params.A() * w_min * w_min) /
                (params.m_eq() * params.m_eq() * w_min * w_min * w_min * w_min + (params.b_eq() * params.b_eq() - 2 * params.m_eq() * params.k_eq()) * w_min * w_min + params.k_eq() * params.k_eq());
                
                // maximum when w is at resonance 
                double w_max = 180 * 2 * 3.14159; //(in radians)
                const T& w_Gw_squared_max = (params.A() * params.A() * w_max * w_max) /
                (params.m_eq() * params.m_eq() * w_max * w_max * w_max * w_max + (params.b_eq() * params.b_eq() - 2 * params.m_eq() * params.k_eq()) * w_max * w_max + params.k_eq() * params.k_eq());
                
                // minimum when w_Gw_squared_min, V_avg = 160, V_dif = 0
                const T& F_t_min = scale_factor.at(0) * 0.5 * rho_B_Cl * w_Gw_squared_min * (160 * 160 + 0 * 0);
                
                // maximum when w_Gw_squared_max, V_avg = 200, V_dif = 30
                const T& F_t_max = scale_factor.at(0) * 0.5 * rho_B_Cl * w_Gw_squared_max * (200 * 200 + 30 * 30);
                
                // minimum when w_Gw_squared_max, V_avg = 160, V_dif = -30
                const T& tau_alpha_min = scale_factor.at(1) * params.r_cp() * rho_B_Cl * w_Gw_squared_max * (200 * (-30));
                
                // maximum when w_Gw_squared_max, V_off = 160, V_dif = 30
                const T& tau_alpha_max = scale_factor.at(1) * params.r_cp() * rho_B_Cl * w_Gw_squared_max * (200 * (30));
                
                // minimum when F_t_max, V_dif = -30
                const T& tau_beta_min = scale_factor.at(2) * params.r_cp() * (-30) * G0 * F_t_max/scale_factor.at(0);
                
                // maximum when F_t_max, V_dif = 30
                const T& tau_beta_max = scale_factor.at(2) * params.r_cp() * (30) * G0 * F_t_max/scale_factor.at(0);
                
                // fix at 0 (assumes kappa = 0.5)
                const T& tau_gamma_min = 0;
                const T& tau_gamma_max = 0;
                
                // scale
                /*
                std::vector<double> scale_factor = {1e4, 1e6, 1e6, 1e6};
                F_t_min = F_t_min * scale_factor.at(0);
                F_t_max = F_t_max * scale_factor.at(0);
                tau_alpha_min = tau_alpha_min * scale_factor.at(1);
                tau_alpha_max = tau_alpha_max * scale_factor.at(1);
                tau_beta_min = tau_beta_min * scale_factor.at(2);
                tau_beta_max = tau_beta_max * scale_factor.at(2);
                tau_gamma_min = tau_gamma_min * scale_factor.at(3);
                tau_gamma_max = tau_gamma_max * scale_factor.at(3); */
                
                // return above values
                std::vector<T> input = {F_t_min, F_t_max, tau_alpha_min, tau_alpha_max, tau_beta_min, tau_beta_max, tau_gamma_min, tau_gamma_max};
                return input;
            }
        }
    }
}

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(class ::drake::examples::robobee::RobobeePlant)
