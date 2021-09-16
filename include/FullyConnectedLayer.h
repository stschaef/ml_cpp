#pragma once
#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#include <Layer.h>

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
                        scalar learning_rate);

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
    void initialize_weights();

    vector<vector<scalar>> weights;
    vector<scalar> biases;
    
    scalar learning_rate;
};

#endif