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
    // n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(28, 28, 4, 0, 3, lr))); //output size 28 - 3 + 1 = 26
    // n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(26, 26, 4, 2)));
    // n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(13, 13, 4, 0, 5, lr))); //13 - 5 + 1 = 9
    // n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(9, 9, 4, 3)));
    // n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(3 * 3 * 4, 32, lr)));
    // n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(32, hyp_tan, hyp_tan_der)));
    // n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(32, 10, lr)));

    n.load_weights("/home/stschaef/ml_cpp/data/mnist_weights.txt");

    vector<vector<scalar>> X_train_before, Y_train_before, X_test, Y_test, X_train, Y_train;

    for (int i = 0; i < 10; i++) {
        vector<string> test_paths = mnist_test(i); 

        // make categorical labels
        vector<scalar> label(10, 0);
        label[i] = 1;
        for (auto p : test_paths) {
            X_test.push_back(flatten_mnist_image(p, 4));
            Y_test.push_back(label);
        }
    }

    cout << n.test(X_test, Y_test) << endl;
    
    return 0;
}