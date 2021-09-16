/*
 * Data sourced from https://www.kaggle.com/alessiocorrado99/animals10
 * Got rid of spider pictures and non .jpeg files
 * Then resized all to (300,200)
 *
*/
#include <iostream>
#include <NeuralNetwork.h>
#include "matplotlibcpp.h"
#include "utils.h"

using namespace std;
namespace plt = matplotlibcpp;

int main()
{   
    vector<scalar> image_vector = flatten_mnist_image("/home/stschaef/ml_cpp/data/mnist/testing/0/3.jpg", 8);
    scalar lr = 0.1;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(28, 28, 1, 0, 2, lr))); //output size 28 - 2 + 1 = 27
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(27, 27, 1, 0, 3, lr))); //27 - 3 + 1 = 25
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(25, 25, 1, 0, 2, lr))); //25 - 2 + 1 = 24
    n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(24, 24, 0, 2)));
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(12, 12, 1, 0, 3, lr))); //25 - 2 + 1 = 24
    n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(10, 10, 1, 2)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(100, 64, lr)));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(64, hyp_tan, hyp_tan_der)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(64, 32, lr)));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(32, hyp_tan, hyp_tan_der)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(32, 16, lr)));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(16, hyp_tan, hyp_tan_der)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(16, 10, lr)));

    vector<scalar> a = n.predict(image_vector);
    
    // vector<vector<scalar>> X;
    // vector<vector<scalar>> Y;

    // vector<int> indices(X.size());
    // iota(indices.begin(), indices.end(), 0);
    // vector<int> train, test;
    // vector<vector<scalar>> X_train, X_test, Y_train, Y_test;

    // for (int i = 0; i < int(.8 * 150); i++) {
    //     X_train.push_back(X[indices[i]]);
    //     Y_train.push_back(Y[indices[i]]);
    // }
    // for (int i = int(.8 * 150); i < 150; i++) {
    //     X_test.push_back(X[indices[i]]);
    //     Y_test.push_back(Y[indices[i]]);
    // }

    // vector<scalar> epoch_data = n.train(X_train, Y_train, 1000, 8);

    // vector<vector<scalar>> output;
    // for (size_t i = 0; i < X.size(); i++) {
    //     output.push_back(n.predict(X[i]));
    // }
    // int num_right = 0;
    // for (size_t i = 0; i < X_test.size(); i++) {
    //     vector<scalar> output = n.predict(X_test[i]);
    //     scalar biggest = -200;
    //     int biggest_index = -1;
    //     for (size_t j = 0; j < output.size(); j++) {
    //         if (output[j] > biggest) biggest_index = j;
    //     }
    //     if (Y_test[i][biggest_index] == 1) num_right++;
    // }

    // cout << "Accuracy: " << scalar(num_right) / X_test.size() << '\n';

    // vector<int> epochs(1000);
    // iota(epochs.begin(), epochs.end(), 1);

    // plt::plot(epochs, epoch_data);
    // plt::xlabel("Number of Iterations");
    // plt::ylabel("Mean Squared Error");
    // plt::title("Animals training");
    // plt::show();
    // plt::save("plots/mnist_training.pdf");

    n.save_weights("data/mnist_weights.txt");
    return 0;
}