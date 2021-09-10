#include <NeuralNetwork.h>

// ___________FullyConnectedLayer Implementations___________

FullyConnectedLayer::FullyConnectedLayer(
    uint n_in,
    uint n_out,
    scalar learning_rate,
    uint32_t seed) 
    : Layer(n_in, n_out),
      rng(mt19937_64(seed)),
      weights(vector<vector<scalar>>(n_inputs, vector<scalar>(n_outputs))),
      learning_rate(learning_rate) 
{     
    initialize_weights(); 
}

void FullyConnectedLayer::initialize_weights() 
{
    // Use Xavier weight initialization 
    // Uniform distribution in [-1/sqrt(n_inputs), 1/sqrt(n_inputs)]
   
    double square_root_bound = 1.0 / sqrt(n_inputs);

    std::uniform_real_distribution<double> unif(-square_root_bound,
                                                square_root_bound);
    
    for (uint i = 0; i < n_inputs; i++) {
        for (uint j = 0; j < n_outputs; j++) {
            weights[i][j] = unif(rng);
        }
    }

    for (uint i = 0; i < n_outputs; i++) {
        biases[i] = unif(rng);
    }
}

vector<scalar> FullyConnectedLayer::forward(vector<scalar> input) 
{
    vector<scalar> output(n_outputs);
    for (uint i = 0; i < n_outputs; i++) {
        output[i] = biases[i];
        for (uint j = 0; j < n_inputs; j++) {
            output[i] += input[i] * weights[i][j];
        }
    }
    return output;
}

vector<scalar> FullyConnectedLayer::backward(
    vector<scalar> input,
    vector<scalar> output_error) 
{
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

    vector<scalar> input_error(n_inputs, 0);

    for (uint i = 0; i < n_inputs; i++) {
        for (uint j = 0; j < n_outputs; j++) {
            input_error[i] += output_error[j] * weights[i][j];
            weights[i][j] -= learning_rate * output_error[j] * input[i];
        }
    }

    for (uint i = 0; i < n_outputs; i++) {
        biases[i] -= learning_rate * output_error[i];
    }
    
    return input_error;
}

// ___________ActivationFunctionLayer Implementations___________
ActivationFunctionLayer::ActivationFunctionLayer(
    uint n_in,
    uint n_out, 
    function<scalar(scalar)> activation,
    function<scalar(scalar)> activation_der) 
    : Layer(n_in, n_out),
      activation(activation),
      activation_der(activation_der) {}

vector<scalar> ActivationFunctionLayer::forward(vector<scalar> input) 
{
    vector<scalar> output(n_outputs, 0);
    for (uint i = 0; i < n_outputs; i++) {
        output[i] = activation(input[i]);
    }

    return output;
}

vector<scalar> ActivationFunctionLayer::backward(vector<scalar> input,
    vector<scalar> output_error)
{
    // Nothing to learn, so just return input error
    vector<scalar> input_error(n_outputs, 0);
    for (uint i = 0; i < n_inputs; i ++) {
        input_error[i] = activation_der(input[i]) * output_error[i];
    }

    return input_error;
}                          


// ___________NeuralNetwork Implementations___________

NeuralNetwork::NeuralNetwork(
    uint input_size,
    uint output_size, 
    vector<uint> topology_in,
    scalar learning_rate,
    function<scalar(scalar, vector<scalar>, vector<scalar>)> loss,
    function<vector<scalar>(vector<scalar>, vector<scalar>)> loss_der) 
    : input_size(input_size),
      output_size(output_size),
      topology(topology_in),
      learning_rate(learning_rate),
      loss(loss),
      loss_prime(loss_der) 
{        
    generate_seeds();
    layers.push_back(new FullyConnectedLayer(
        input_size,
        topology[0],
        learning_rate,
        seeds[0]));
    for (uint i = 1; i < topology.size() - 1; i++) {
        layers.push_back(new FullyConnectedLayer(
            topology[i - 1], 
            topology[i],
            learning_rate,
            seeds[i]));
    }
    layers.push_back(new FullyConnectedLayer(
        topology[topology.size() - 1],
        output_size,
        learning_rate,
        seeds[seeds.size() - 1]));
}

void NeuralNetwork::generate_seeds() {
    seeds.resize(topology.size() + 2);
    std::seed_seq ss{8, 6, 7, 5, 3, 0, 9}; // TODO: change this from hard-coded
    ss.generate(seeds.begin(), seeds.end());
}

