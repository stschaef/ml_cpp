#include <iostream>
#include <NeuralNetwork.h>
#include <emscripten/emscripten.h>


using namespace std;

extern "C" {
EMSCRIPTEN_KEEPALIVE
scalar * predict(scalar * inp)
{   
    scalar * ans = (scalar*)calloc(10, sizeof(scalar));
    scalar lr = 0.1;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.load_weights("/home/stschaef/ml_cpp/mnist_weights_97.txt");

    vector<scalar> preds = n.predict(vector<scalar>(inp, inp + sizeof(inp) / sizeof(inp[0])));
    copy(preds.begin(), preds.end(), ans);
    return ans;
}
}