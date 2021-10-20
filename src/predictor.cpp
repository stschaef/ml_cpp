#include <iostream>
#include <NeuralNetwork.h>
// #include <emscripten.h>


using namespace std;

// EMSCRIPTEN_KEEPALIVE

extern "C" {
int predict()
{   
    scalar lr = 0.1;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.load_weights("/home/stschaef/ml_cpp/mnist_weights_97.txt");

    cout << "Hello World!" << '\n';
    
    return 0;
}}