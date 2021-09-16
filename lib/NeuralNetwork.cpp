#include <NeuralNetwork.h>
// #include <Layer.h>
// #include <FullyConnectedLayer.h>
// #include
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
    uint num_epochs,
    uint batch_size)
{
    vector<scalar> epoch_data;
    size_t num_samples = input_data.size();

    vector<int> indices(num_samples);
    iota(indices.begin(), indices.end(), 0);

    for (uint i = 0; i < num_epochs; i++) {
        random_shuffle(indices.begin(), indices.end());
        scalar error = 0.0;

        uint j = 0;
        uint k = 0;

        while (j < num_samples) {
            vector<int> mini_batch;
            while (k < batch_size) {
                j++;
                k++;
                mini_batch.push_back(j % num_samples);
            }
            k = 0;

            vector<scalar> batch_grad(input_data[0].size());
        
            // average the gradient over batch
            for (auto idx : mini_batch) {
                vector<scalar> output = predict(input_data[idx]);
                vector<scalar> output_error = loss_der(actual[idx], output);
                error += loss(actual[idx], output) * (1.0 / batch_size);

                for (size_t z = 0; z < output_error.size(); z++) {
                    batch_grad[z] += output_error[z] * (1.0 / batch_size);
                }
            }
            for (int l = layers.size() - 1; l >= 0; l--) {
                batch_grad = layers[l]->backward(batch_grad);
            }
        }
        
        error = error / num_epochs;
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
                // FullyConnectedLayer
                case 'a': 
                {
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
                // ActivationFunctionLayer
                case 'b':
                {
                    out << layer_type << '\n';
                    break;
                }
                // MaxPoolingLayer
                case 'c':
                {
                    out << layer_type << '\n';  
                    break;
                }
                // AveragePoolingLayer
                case 'd':
                {
                    out << layer_type << '\n'; 
                    break;
                }
                // ConvolutionLayer
                case 'e':
                {
                    vector<vector<vector<scalar>>> kers = (dynamic_cast<ConvolutionLayer*> (layers[i].get()))->get_kernels();
                    out << "kernels\n";
                    for (size_t c = 0; c < kers.size(); c++) {
                        out << "channel\n";
                        for (size_t j = 0; j < kers[0].size(); j++) {
                            for (size_t k = 0; k < kers[0][0].size(); k++) {
                                out << kers[c][j][k] << " ";
                            }
                        }
                        out << '\n';
                    }
                    break;
                }
            }
        }
    }
    out.close();
}