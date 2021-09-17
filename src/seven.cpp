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

    n.load_weights("data/mnist_weights_91.txt");
    
    return 0;
}