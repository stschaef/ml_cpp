#include <iostream>
#include <NeuralNetwork.h>
// #include <csignal>
#include "matplotlibcpp.h"
#include "utils.h"

using namespace std;
namespace plt = matplotlibcpp;

// TODO make image data class for data augmentation
// implement affine transformations for these, scale, rotate, translate
// then apply each with small parameters

int main()
{   
    uint total_epochs = 40;
    uint batch_size = 8;
    uint num_channels = 5;
    scalar lr = 0.5;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(28, 28, num_channels, 0, 3, lr))); //output size 28 - 3 + 1 = 26, in 6 channels
    n.add(make_shared<ReLULayer>(ReLULayer(26 * 26 * num_channels))); 
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(26, 26, num_channels, 0, 5, lr))); //output size 26 - 5 + 1 = 22
    n.add(make_shared<TanhLayer>(TanhLayer(22 * 22 * num_channels)));
    n.add(make_shared<AveragePoolingLayer>(AveragePoolingLayer(22, 22, num_channels, 2)));
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(11, 11, num_channels, 0, 5, lr))); //11 - 5 + 1 =7
    n.add(make_shared<ReLULayer>(ReLULayer(7 * 7 * num_channels))); 
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(7 * 7 * num_channels, 64 , lr)));
    n.add(make_shared<TanhLayer>(TanhLayer(64)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(64, 10, lr)));

    vector<vector<scalar>> X_train_before, Y_train_before, X_test, Y_test, X_train, Y_train;

    for (int i = 0; i < 10; i++) {
        vector<string> train_paths = mnist_train(i); 
        vector<string> test_paths = mnist_test(i); 

        // make categorical labels
        vector<scalar> label(10, 0);
        label[i] = 1;

        for (auto p : train_paths) {
            X_train_before.push_back(flatten_mnist_image(p, num_channels));
            Y_train_before.push_back(label);
        }

        for (auto p : test_paths) {
            X_test.push_back(flatten_mnist_image(p, num_channels));
            Y_test.push_back(label);
        }
    }
    
    // Randomize order of training
    vector<int> indices(X_train_before.size());
    iota(indices.begin(), indices.end(), 0);
    
    random_shuffle(indices.begin(), indices.end());

    // Could maybe do this in place with a permutation, but this is easy
    for (size_t i = 0; i < indices.size(); i++) {
        X_train.push_back(X_train_before[indices[i]]);
        Y_train.push_back(Y_train_before[indices[i]]);
    }

    vector<scalar> testing_accuracy;
    vector<scalar> epoch_data = n.train(X_train,
                                        Y_train,
                                        total_epochs,
                                        batch_size,
                                        X_test,
                                        Y_test,
                                        testing_accuracy);

    n.save_weights("data/mnist_weights.txt");

    vector<int> epochs(total_epochs);
    iota(epochs.begin(), epochs.end(), 1);

    plt::plot(epochs, testing_accuracy);
    plt::xlabel("Number of Epochs");
    plt::ylabel("Test Set Accuracy/MSE");

    plt::plot(epochs, epoch_data);
    
    plt::title("MNIST Handwriting: Test Accuracy and Training Loss (MSE)");
    // plt::show();
    plt::save("plots/mnist_training.pdf");

    return 0;
}