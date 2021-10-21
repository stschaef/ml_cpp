#include <iostream>
#include <NeuralNetwork.h>
#include <emscripten/emscripten.h>


using namespace std;

extern "C" {
EMSCRIPTEN_KEEPALIVE
vector<scalar> predict(vector<scalar> input_vec)
{   
    scalar lr = 0.1;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.load_weights("/home/stschaef/ml_cpp/mnist_weights_97.txt");

    return(n.predict(input_vec));
}
}