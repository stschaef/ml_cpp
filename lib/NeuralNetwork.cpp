#include <NeuralNetwork.h>

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
    int num_epochs,
    int batch_size,
    vector<vector<scalar>>& X_test,
    vector<vector<scalar>>& Y_test,
    vector<scalar>& testing_accuracy)
{
    vector<scalar> epoch_data;
    size_t num_samples = input_data.size();

    vector<int> indices(num_samples);
    iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < num_epochs; i++) {
        random_shuffle(indices.begin(), indices.end());
        scalar error = 0.0;

        int j = 0;
        int k = 0;

        while (j < int(num_samples)) {
            vector<int> mini_batch;
            while (k < batch_size) {
                j++;
                k++;
                mini_batch.push_back(indices[j] % num_samples);
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
        
        error = error / (num_samples / batch_size);
        cout << "Epoch " << i << "\tError: " << error << '\n';
        // printf("Epoch: %d \tError: %f \n", i, error);
        epoch_data.push_back(error);

        scalar accuracy = this->test(X_test, Y_test);
        cout << "Accuracy: " << accuracy << '\n';
        // printf("Accuracy: %f \n", accuracy);
        testing_accuracy.push_back(accuracy);
    }
    return epoch_data;
}

scalar NeuralNetwork::test(vector<vector<scalar>>& X_test,
                           vector<vector<scalar>>& Y_test) 
{
    int num_right = 0;
    for (size_t i = 0; i < X_test.size(); i++) {
        vector<scalar> test_output = this->predict(X_test[i]);
        int biggest_index = 0;
        for (size_t j = 0; j < test_output.size(); j++) {
            if (test_output[j] > test_output[biggest_index]) {
                biggest_index = j;
            }
        }
        if (Y_test[i][biggest_index] == 1) num_right++;
    }

    return scalar(num_right) / scalar(X_test.size());
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
                    out << layer_type << " " << layers[i]->get_n_inputs() << " " << layers[i]->get_n_outputs() << "\n";

                    break;
                }
                // MaxPoolingLayer
                case 'c':
                {
                    out << layer_type << " " << layers[i]->get_n_inputs() << " " << layers[i]->get_n_outputs() << "\n";
                    break;
                }
                // AveragePoolingLayer
                case 'd':
                {
                    out << layer_type << " " << layers[i]->get_n_inputs() << " " << layers[i]->get_n_outputs() << "\n";
                    break;
                }
                // ConvolutionLayer
                case 'e':
                {
                    out << layer_type << " " << layers[i]->get_n_inputs() << " " << layers[i]->get_n_outputs() << "\n";

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

void NeuralNetwork::load_weights(string in_filename)
{
    ifstream in_file(in_filename);
    for (string line; getline(in_file, line); ) {
        cout << line << endl;
        istringstream iss(line);
        vector<string> words;
        do
        {
            string word;
            iss >> word;
            words.push_back(word);
        } while (iss);
        
        if (words.empty()) return;

        // if (words[0] == "a") {

        // }
        // else if (words[0] == "b") {

        // }
        // else if (words[0] == "c") {
            
        // }
        // else if (words[0] == "d") {
            
        // }
        // else if (words[0] == "e") {
            
        // }
        // else if (words[0] == "kernels") {
            
        // }
        // else if (words[0] == "channel") {
            
        // }
        // else if (words[0] == "weights") {
            
        // }
        // else if (words[0] == "biases") {
            
        // }

        return;
    }
}