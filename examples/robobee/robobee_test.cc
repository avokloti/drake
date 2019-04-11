#include <memory>
#include <iostream>
#include <fstream>

#include <gflags/gflags.h>

#include "drake/systems/framework/system.h"
#include "drake/systems/framework/context.h"
#include "drake/examples/robobee/robobee_plant.h"
#include "drake/math/wrap_to.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
    namespace examples {
        namespace robobee {
            namespace {
                
                int do_main(int argc, char* argv[]) {
                
                    // make robobee plant
                    RobobeePlant<double>* plant = new RobobeePlant<double>();
                    auto context_ptr = plant->CreateDefaultContext();
                    
                    // print number of states and inputs
                    std::cout << "Number of states: " << context_ptr->get_num_total_states()<< std::endl;
                    std::cout << "Number of inputs: " << plant->get_input_port().size() << "\n" << std::endl;
                    
                    // --- calculate and print dynamics ---
                    
                    // define voltages and low-level input
                    double V_avg = 180;
                    double V_diff = 10;
                    double V_off = 10;
                    double w = 160;
                    
                    // calculate high-level input values (thrust force and torques)
                    std::vector<double> inputs = plant->SetInputFromVoltage(*context_ptr, V_avg, V_diff, V_off, w);
                    
                    // define specific state and input values
                    Eigen::MatrixXd state_value = Eigen::VectorXd::Zero(12);
                    Eigen::Map<Eigen::VectorXd> input_value(inputs.data(), 4);
                    
                    // set context state value to a specific state
                    context_ptr->get_mutable_continuous_state().SetFromVector(state_value);
                    
                    // set context input value to a specific input
                    auto input_port_value = &context_ptr->FixInputPort(0, plant->AllocateInputVector(plant->get_input_port()));
                    input_port_value->systems::FixedInputPortValue::GetMutableVectorData<double>()->SetFromVector(input_value);
                    
                    // prepare derivative vector and calculate derivatives
                    std::unique_ptr<systems::ContinuousState<double> > continuous_state(plant->AllocateTimeDerivatives());
                    plant->CalcTimeDerivatives(*context_ptr, continuous_state.get());
                    Eigen::MatrixXd derivative = continuous_state->CopyToVector();
                    
                    // print
                    std::cout << "states are: " << std::endl;
                    std::cout << state_value << "\n" << std::endl;
                    
                    std::cout << "inputs are: " << std::endl;
                    std::cout << input_value << "\n" << std::endl;
                    
                    std::cout << "derivatives are: " << std::endl;
                    std::cout << derivative << "\n" << std::endl;
                    
                    return 0;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    return drake::examples::robobee::do_main(argc, argv);
}
