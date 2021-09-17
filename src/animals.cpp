/*
 * Data sourced from https://www.kaggle.com/alessiocorrado99/animals10
 * Got rid of spider pictures and non .jpeg files
 * Then resized all to (300,200)
 *
*/
#include <iostream>
#include <NeuralNetwork.h>
#include <opencv4/opencv2/opencv.hpp>  
#include "matplotlibcpp.h"
#include "utils.h"

using namespace std;
namespace plt = matplotlibcpp;

int main()
{   
    vector<scalar> image_vector = flatten_animals_image("/home/stschaef/ml_cpp/data/animals_resized/cat/1.jpeg");

    scalar lr = 0.1;

    // TODO: fix this
    // This is laughably too large, and the layers are overflowing because of it
    // even without the overflow, it is questionableif i can run this
    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(300, 200, 3, 0, 5, lr)));
    n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(300 - 5 + 1, 200 - 5 + 1, 3, 2)));
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer((300 -5 + 1)/2, (200 -5 + 1)/2, 3, 0, 6, lr)));
    n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer((300 -5 + 1)/2 - 6 + 1, (200 - 5 + 1)/2 - 6 + 1, 3, 2)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(143 * 93, 2000, lr)));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(2000, hyp_tan, hyp_tan_der)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(2000, 90, lr)));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(90, hyp_tan, hyp_tan_der)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(90, 9, lr)));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(9, hyp_tan, hyp_tan_der)));

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
    // plt::save("plots/animals_training.pdf");

    n.save_weights("data/animal_weights.txt");
    return 0;
}