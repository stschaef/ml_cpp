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
    uint num_epochs,
    uint batch_size,
    vector<vector<scalar>>& X_test,
    vector<vector<scalar>>& Y_test,
    vector<scalar>& testing_accuracy)
{
    vector<scalar> epoch_data;
    size_t num_samples = input_data.size();

    if (num_samples == 0) return {};

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
        epoch_data.push_back(error);

        scalar accuracy = this->test(X_test, Y_test);
        cout << "Accuracy: " << accuracy << '\n';
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
                    out << layer_type << " " 
                        << dynamic_cast<MaxPoolingLayer*>(layers[i].get())->get_width() << " " 
                        << dynamic_cast<MaxPoolingLayer*>(layers[i].get())->get_height() << " " 
                        << dynamic_cast<MaxPoolingLayer*>(layers[i].get())->get_num_channels() << " " 
                        << dynamic_cast<MaxPoolingLayer*>(layers[i].get())->get_pooling_size() << "\n"; 
                    break;
                }
                // AveragePoolingLayer
                case 'd':
                {
                    out << layer_type << " " 
                        << dynamic_cast<AveragePoolingLayer*>(layers[i].get())->get_width() << " " 
                        << dynamic_cast<AveragePoolingLayer*>(layers[i].get())->get_height() << " " 
                        << dynamic_cast<AveragePoolingLayer*>(layers[i].get())->get_num_channels() << " " 
                        << dynamic_cast<AveragePoolingLayer*>(layers[i].get())->get_pooling_size() << "\n"; 
                    break;
                }
                // ConvolutionLayer
                case 'e':
                {
                    out << layer_type << " " 
                        << dynamic_cast<ConvolutionLayer*>(layers[i].get())->get_width() << " " 
                        << dynamic_cast<ConvolutionLayer*>(layers[i].get())->get_height() << " " 
                        << dynamic_cast<ConvolutionLayer*>(layers[i].get())->get_num_channels() << " " 
                        << dynamic_cast<ConvolutionLayer*>(layers[i].get())->get_padding_size() << " " 
                        << dynamic_cast<ConvolutionLayer*>(layers[i].get())->get_kernel_size() << "\n"; 

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
                case 'x':
                {
                    out << layer_type << " " << layers[i]->get_n_inputs() << "\n";
                    break;
                }
                case 'y':
                {
                    out << layer_type << " " << layers[i]->get_n_inputs() << "\n";
                    break;
                }
            }
        }
    }
    out.close();
}

void NeuralNetwork::load_weights(string in_filename)
{
    cout << in_filename;
    ifstream in_file(in_filename);

    char layer_type;
    uint channel_idx = -1;
    uint weight_row = 0;
    bool bias = false;

    for (string line; getline(in_file, line); ) {
        istringstream iss(line);
        vector<string> words;
        do
        {
            string word;
            iss >> word;
            words.push_back(word);
        } while (iss);
    
        if (words.empty()) return;

        // FullyConnectedLayer      : 'a';
        // ActivationFunctionLayer  : 'b';
        // MaxPoolingLayer          : 'c';
        // AveragePoolingLayer      : 'd';
        // ConvolutionLayer         : 'e';
        // TanhLayer                : 'x';
        // ReLULayer                : 'y';

        if (words[0] == "a") {
            layer_type = 'a';
            add(make_shared<FullyConnectedLayer>(FullyConnectedLayer((uint) stoi(words[1]), (uint) stoi(words[2]), learning_rate)));
        }
        else if (words[0] == "b") {
            continue;
            // 'b' shouldnt happen
            // we use TanhLayer and ReLULayer in place of ActivationFunctionLayer when saving
            // Saving custom activation functions not happening right now
        }
        else if (words[0] == "x") {
            layer_type = 'x';

            add(make_shared<TanhLayer>(TanhLayer((uint) stoi(words[1]))));
        }
        else if (words[0] == "y") {
            layer_type = 'y';
            add(make_shared<ReLULayer>(ReLULayer((uint) stoi(words[1]))));
        }
        else if (words[0] == "c") {
            layer_type = 'c';
            add(make_shared<MaxPoolingLayer>(MaxPoolingLayer((uint) stoi(words[1]), (uint) stoi(words[2]), (uint) stoi(words[3]), (uint) stoi(words[4]))));
        }
        else if (words[0] == "d") {
            layer_type = 'd';
            add(make_shared<AveragePoolingLayer>(AveragePoolingLayer((uint) stoi(words[1]), (uint) stoi(words[2]), (uint) stoi(words[3]), (uint) stoi(words[4]))));
        }
        else if (words[0] == "e") {
            layer_type = 'e';
            add(make_shared<ConvolutionLayer>(ConvolutionLayer((uint) stoi(words[1]), (uint) stoi(words[2]), (uint) stoi(words[3]), (uint) stoi(words[4]), (uint) stoi(words[5]), learning_rate)));
        }
        else if (words[0] == "kernels") {
            channel_idx = -1;
            continue;
        }
        else if (words[0] == "biases") {
            bias = true;
            weight_row = 0;
        }
        else if (words[0] == "weights") {
            bias = false;
        }
        else if (words[0] == "channel") {
            channel_idx++;
        }
        // numerical data
        else {
            switch (layer_type)
            {
            case 'a': {
                if (!bias) {
                    for (size_t i = 0; i < words.size() - 1; i++) {
                        dynamic_cast<FullyConnectedLayer*>(layers[layers.size() - 1].get())->set_weight_at(weight_row, i, stod(words[i]));
                    }
                    weight_row++;
                }
                else {
                    for (size_t i = 0; i < words.size() - 1; i++) {
                        dynamic_cast<FullyConnectedLayer*>(layers[layers.size() - 1].get())->set_bias_at(i, stod(words[i]));
                    }
                }
                break;
            }
            case 'b':
                break;
            
            case 'c':
                break;
            case 'd':
                break;
            case 'e': {
                uint ker_size(sqrt(words.size() - 1));
                for (uint i = 0; i < words.size() - 1; i++) {
                    dynamic_cast<ConvolutionLayer*>(layers[layers.size() - 1].get())->set_kernel_at(channel_idx, i / ker_size, i % ker_size, stod(words[i]));
                }
                break;
            }
            case 'x':
                break;
            case 'y':
                break;
            default:
                break;
            }
        }
    }

        return;
}