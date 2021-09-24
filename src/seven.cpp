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
    n.load_weights("/home/stschaef/ml_cpp/mnist_weights_91.txt");

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