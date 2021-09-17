#pragma once
#ifndef ACTIVATIONFUNCTIONLAYER_H_
#define ACTIVATIONFUNCTIONLAYER_H_

#include <Layer.h>

class ActivationFunctionLayer : public Layer {
public:
    // n_in == n_out in this layer
    ActivationFunctionLayer(int n,
                            function<scalar(scalar)> activation,
                            function<scalar(scalar)> activation_der);

    vector<scalar> forward(vector<scalar> input);
    vector<scalar> backward(vector<scalar> output_error);
private:
    function<scalar(scalar)> activation;
    function<scalar(scalar)> activation_der;
};

// ___________Activation Functions___________
inline scalar hyp_tan(scalar x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

inline scalar hyp_tan_der(scalar x)
{
    return 1 - pow(hyp_tan(x), 2);
}

inline scalar relu(scalar x)
{
    return max(scalar(0.0), x);
}

inline scalar relu_der(scalar x)
{
    return (x > 0) ? 1 : 0;
}

#endif