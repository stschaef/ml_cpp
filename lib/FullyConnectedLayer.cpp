#include <FullyConnectedLayer.h>
#include <Layer.h>

// ___________FullyConnectedLayer Implementations___________
FullyConnectedLayer::FullyConnectedLayer(
    uint n_in,
    uint n_out,
    scalar learning_rate) 
    : Layer(n_in, n_out),
      weights(vector<vector<scalar>>(n_inputs, vector<scalar>(n_outputs))),
      biases(vector<scalar>(n_outputs)),
      learning_rate(learning_rate)
{     
    initialize_weights(); 
}

void FullyConnectedLayer::initialize_weights() 
{
    // Use Xavier weight initialization 
    // Uniform distribution in [-1/sqrt(n_inputs), 1/sqrt(n_inputs)]
    rng.seed(seeds[num_seeds_used++]);
   
    scalar square_root_bound = 1.0 / sqrt(n_inputs);

    std::uniform_real_distribution<scalar> unif(-square_root_bound,
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
    in = input;
    vector<scalar> output(n_outputs);
    for (uint j = 0; j < n_outputs; j++) {
        output[j] = biases[j];
        for (uint i = 0; i < n_inputs; i++) {
            output[j] += input[i] * weights[i][j];
        }
    }
    return output;
}

vector<scalar> FullyConnectedLayer::backward(
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
            weights[i][j] -= learning_rate * output_error[j] * in[i];
        }
    }

    for (uint i = 0; i < n_outputs; i++) {
        biases[i] -= learning_rate * output_error[i];
    }
    
    return input_error;
}

void FullyConnectedLayer::set_weight_at(uint i, uint j, scalar val)
{
    weights[i][j] = val;
}

void FullyConnectedLayer::set_bias_at(uint i, scalar val)
{
    biases[i] = val;
}