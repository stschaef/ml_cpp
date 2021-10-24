#include <iostream>
#include <NeuralNetwork.h>
#include <emscripten/emscripten.h>


using namespace std;

extern "C" {
EMSCRIPTEN_KEEPALIVE
scalar predict(void * inp, int i)
{   
    scalar * in = (double *) inp;
    // scalar * ans = (scalar*)calloc(10, sizeof(scalar));
    scalar lr = 0.1;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.load_weights("mnist_weights_97.txt");

    vector<scalar> v(28*28*5);
    cout << "input vec" << '\n';
    for (int i = 0; i < 28*28*5; i++) {
        v[i] = in[i];
        // cout << v[i] << '\n';
    }
    vector<scalar> preds = n.predict(v);
    // copy(preds.begin(), preds.end(), ans);

    cout << "predictions" << "\n";
    for (int i = 0; i < 10; i++) {
        cout << preds[i] << '\n';
    }
    return preds[i];
}
}