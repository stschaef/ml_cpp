#include <ActivationFunctionLayer.h>
#include <Layer.h>

// ___________ActivationFunctionLayer Implementations___________
ActivationFunctionLayer::ActivationFunctionLayer(
    uint n ,
    function<scalar(scalar)> activation,
    function<scalar(scalar)> activation_der) 
    : Layer(n, n),
      activation(activation),
      activation_der(activation_der) {}

vector<scalar> ActivationFunctionLayer::forward(vector<scalar> input) 
{   
    in = input;
    vector<scalar> output(n_outputs, 0);
    for (uint i = 0; i < n_outputs; i++) {
        output[i] = activation(input[i]);
    }

    return output;
}

vector<scalar> ActivationFunctionLayer::backward(
    vector<scalar> output_error)
{
    // Nothing to learn, so just return input error
    vector<scalar> input_error(n_outputs, 0);
    for (uint i = 0; i < n_inputs; i++) {
        input_error[i] = activation_der(in[i]) * output_error[i];
    }

    return input_error;
}                          

ReLULayer::ReLULayer(uint n) : ActivationFunctionLayer(n, relu, relu_der) {}

TanhLayer::TanhLayer(uint n) : ActivationFunctionLayer(n, hyp_tan, hyp_tan_der) {}