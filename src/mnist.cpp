#include <iostream>
#include <NeuralNetwork.h>
// #include <csignal>
#include "matplotlibcpp.h"
#include "utils.h"

using namespace std;
namespace plt = matplotlibcpp;

int main()
{   
    scalar lr = 0.1;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    // n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(28, 28, 8, 0, 2, lr))); //output size 28 - 2 + 1 = 27
    // n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(27, 27, 8, 0, 3, lr))); //27 - 3 + 1 = 25
    // n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(25, 25, 8, 0, 2, lr))); //25 - 2 + 1 = 24
    // n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(24, 24, 8, 2)));
    // n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(12, 12, 8, 0, 3, lr))); //25 - 2 + 1 = 24
    // n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(10, 10, 8, 2)));
    // n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(200, 128, lr)));
    // n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(128, hyp_tan, hyp_tan_der)));
    // n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(128, 64, lr)));
    // n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(64, hyp_tan, hyp_tan_der)));
    // n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(64, 32, lr)));
    // n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(32, hyp_tan, hyp_tan_der)));
    // n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(32, 10, lr)));

    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(28, 28, 1, 0, 3, lr))); //output size 28 - 2 + 1 = 27
    n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(26, 26, 1, 2)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(169, 32, lr)));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(32, hyp_tan, hyp_tan_der)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(32, 10, lr)));

    vector<vector<scalar>> X_train_before, Y_train_before, X_test, Y_test, X_train, Y_train;

    for (int i = 0; i < 10; i++) {
        vector<string> train_paths = mnist_train(i); 
        vector<string> test_paths = mnist_test(i); 

        // make categorical labels
        vector<scalar> label(10, 0);
        label[i] = 1;

        for (auto p : train_paths) {
            X_train_before.push_back(flatten_mnist_image(p, 1));
            Y_train_before.push_back(label);
        }

        for (auto p : test_paths) {
            X_test.push_back(flatten_mnist_image(p, 1));
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
    vector<scalar> epoch_data = n.train(X_train, Y_train, 2, 64, X_test, Y_test, testing_accuracy);

    vector<int> epochs(2);
    iota(epochs.begin(), epochs.end(), 1);

    plt::plot(epochs, epoch_data);
    plt::xlabel("Number of Epochs");
    plt::ylabel("Mean Squared Error");

    plt::plot(epochs, testing_accuracy);
    plt::xlabel("Number of Epochs");
    plt::ylabel("Accuracy");
    
    plt::title("MNIST Handwriting Training");
    // plt::show();
    plt::save("plots/mnist_training.pdf");

    n.save_weights("data/mnist_weights.txt");
    return 0;
}