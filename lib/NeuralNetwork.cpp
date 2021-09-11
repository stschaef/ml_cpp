#include <NeuralNetwork.h>

// ___________Layer Implementations___________
char Layer::get_layer_type()
{
    // Use this for storing weight information into files
    // Encode each layer type with a char
    if (dynamic_cast<FullyConnectedLayer*>(this)) return 'f';
    if (dynamic_cast<ActivationFunctionLayer*>(this)) return 'a';
    return 'x';
}

// ___________FullyConnectedLayer Implementations___________
FullyConnectedLayer::FullyConnectedLayer(
    uint n_in,
    uint n_out,
    scalar learning_rate,
    uint32_t seed) 
    : Layer(n_in, n_out),
      rng(mt19937_64(seed)),
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
    for (uint i = 0; i < n_inputs; i ++) {
        input_error[i] = activation_der(in[i]) * output_error[i];
    }

    return input_error;
}                          


// ___________NeuralNetwork Implementations___________

NeuralNetwork::NeuralNetwork(
    scalar learning_rate,
    function<scalar(vector<scalar>, vector<scalar>)> loss,
    function<vector<scalar>(vector<scalar>, vector<scalar>)> loss_der) 
    : learning_rate(learning_rate),
      loss(loss),
      loss_der(loss_der) {}

vector<scalar> NeuralNetwork::predict(vector<scalar> input) 
{
    vector<scalar> output = input;
    for (size_t i = 0; i < layers.size(); i++) {
        output = layers[i]->forward(output);
    }
    return output;
}

vector<scalar> NeuralNetwork::train(
    vector<vector<scalar>>& input_data,
    vector<vector<scalar>>& actual,
    uint num_epochs)
{
    vector<scalar> epoch_data;

    for (uint i = 0; i < num_epochs; i++) {
        scalar error = 0.0;
        for (size_t j = 0; j < input_data.size(); j++) {
            vector<scalar> output = predict(input_data[j]);
            error += loss(actual[j], output);

            vector<scalar> output_error = loss_der(actual[j], output);
            for (int k = layers.size() - 1; k >= 0; k--) {
                output_error = layers[k]->backward(output_error);
            }

        }
        error = error / input_data.size();
        cout << "Epoch " << i << "\tError: " << error << '\n';
        epoch_data.push_back(error);
    }
    return epoch_data;
}

void NeuralNetwork::add(shared_ptr<Layer> l)
{
    layers.push_back(l);
}

void NeuralNetwork::save_weights(string out_filename)
{
    ofstream out;
    out.open(out_filename);
    if (out.is_open()) {
        for (size_t i = 0; i < layers.size(); i++) {
            char layer_type = layers[i]->get_layer_type();
            switch(layer_type) {
                case 'a':
                    out << layer_type << '\n';
                    break;
                case 'f':
                    vector<vector<scalar>> layer_weights = (dynamic_cast<FullyConnectedLayer*> (layers[i].get()))->get_weights();
                    vector<scalar> layer_biases = (dynamic_cast<FullyConnectedLayer*> (layers[i].get()))->get_biases();

                    out << layer_type << " " << layers[i]->get_n_inputs() << " " << layers[i]->get_n_outputs() << "\n";

                    out << "weights\n";
                    for (size_t j = 0; j < layer_weights.size(); j++) {
                        for (size_t k = 0; k < layer_weights[0].size(); k++) {
                            out << layer_weights[j][k] << " ";
                        }
                        out << '\n';
                    }
                     out << "biases\n";
                    for (size_t j = 0; j < layer_biases.size(); j++) {
                        out << layer_biases[j] << " ";
                    }
                    out << '\n';
                    break;
            }
        }
    }
    out.close();
}