#include <iostream>
#include <NeuralNetwork.h>


using namespace std;

// Test on my own handwritten 7
int main()
{   
    scalar lr = 0.1;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.load_weights("/home/stschaef/ml_cpp/mnist_weights_97.txt");

    // auto a = flatten_mnist_image("/home/stschaef/ml_cpp/data/seven.jpg", 5);

    // auto b = n.predict(a);

    // for (auto c : b) {
    //     cout << c << '\n';
    // }
    cout << "Hello World!" << '\n';
    
    return 0;
}