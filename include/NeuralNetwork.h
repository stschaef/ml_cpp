#pragma once
#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>

typedef float scalar;
using namespace std;

class Layer {
public:
    Layer(uint n_in, uint n_out) : n_inputs(n_in), n_outputs(n_out) {}

    virtual vector<scalar> forward(vector<scalar> input) = 0;   
    virtual vector<scalar> backward(vector<scalar> output_error) = 0;
    vector<scalar> in;

    char get_layer_type();
    inline uint get_n_inputs() {return n_inputs;};
    inline uint get_n_outputs() {return n_outputs;};
protected:
    uint n_inputs;
    uint n_outputs;
};

class FullyConnectedLayer : public Layer {
/* Describe the math behind neural net calculations
 * with the following variables
 *
 * error: E
 * input: X
 * output: Y
 * weights: W
 * biases: B
 */
public:
    FullyConnectedLayer(uint n_in,
                        uint n_out,
                        scalar learning_rate,
                        uint32_t seed);

    vector<scalar> forward(vector<scalar> input);

    /* How backpropagation works:
     *
     * output_error : dE / dY
     * input_error: dE / dX
     * weight_error: dE / dW
     * bias_error: dE / dB
     * 
     * By the chain rule we have, 
     * input_error:
     *   dE / dx_i = \sum_j (dE / dy_j * dy_j / dx_i)
     *   Accumulate over sum:
     *       input_error[i] += output_error[j] * weights[i][j];
     * 
     * weight_error:
     *   dE / dw_ij = \sum_k (dE / dy_k * dy_k / dw_ij)
     *              = dE / dy_j * x_i
     *              by linearity of Y = XW + B
     *   That is,
     *   dE / dW = [dE/dy_j * x_i]
     *   Calculate and update in place:
     *     weights[i][j] -= learning_rate * output_error[j] * input[i];
     * 
     * 
     * bias_error:
     *   dE / db_i = \sum (dE / dy_j * dy_j / db_i)
     *             = dE / dy_i
     *             because dy_j / db_i = kronecker_delta(i, j)
     */
    vector<scalar> backward(vector<scalar> output_error);

    inline vector<vector<scalar>> get_weights() {return weights;}
    inline vector<scalar> get_biases() {return biases;}
private:
    std::mt19937_64 rng;
    void initialize_weights();

    vector<vector<scalar>> weights;
    vector<scalar> biases;
    
    scalar learning_rate;
};

class ActivationFunctionLayer : public Layer {
public:
    // n_in == n_out in this layer
    ActivationFunctionLayer(uint n,
                            function<scalar(scalar)> activation,
                            function<scalar(scalar)> activation_der);

    vector<scalar> forward(vector<scalar> input);
    vector<scalar> backward(vector<scalar> output_error);
private:
    function<scalar(scalar)> activation;
    function<scalar(scalar)> activation_der;
};

class NeuralNetwork {
public:
    NeuralNetwork(scalar learning_rate,
                  function<scalar(vector<scalar>, vector<scalar>)> loss,
                  function<vector<scalar>(vector<scalar>, vector<scalar>)> loss_der);

    vector<scalar> predict(vector<scalar> input);

    vector<scalar> train(vector<vector<scalar>>& input_data,
               vector<vector<scalar>>& actual,
               uint num_epochs,
               uint batch_size);
    void add(shared_ptr<Layer> l);

    void save_weights(string out_filename);
private:
    vector<shared_ptr<Layer>> layers;

    void generate_seeds();
    vector<uint32_t> seeds;

    scalar learning_rate;

    function<scalar(vector<scalar>, vector<scalar>)> loss;
    function<vector<scalar>(vector<scalar>, vector<scalar>)> loss_der;
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


// ___________Loss Functions___________
inline scalar mean_squared_error(vector<scalar> actual, vector<scalar> predicted)
{
    scalar sum = 0;
    int n = actual.size();
    for (int i = 0; i < n; i++) {
        sum += pow(actual[i] - predicted[i], 2);
    }
    sum *= 1.0 / n;
    return sum;
}

inline vector<scalar> mean_squared_error_der(vector<scalar> actual, vector<scalar> predicted)
{   
    int n = actual.size();
    for (int i = 0; i < n; i++) {
        actual[i] = predicted[i] - actual[i];
        actual[i] *= 2.0 / n;
    }

    return actual;
}

#endif