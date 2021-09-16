/*
 * Data sourced from https://www.kaggle.com/alessiocorrado99/animals10
 * Got rid of spider pictures and non .jpeg files
 * Then resized all to (300,200)
 *
*/
#include <iostream>
#include <NeuralNetwork.h>
#include <Python.h>
#include "matplotlibcpp.h"
#include "jpeglib.h"

using namespace std;
namespace plt = matplotlibcpp;

int main()
{   
    MaxPoolingLayer p(4, 4, 1, 2);
    vector<scalar> a = p.forward({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    vector<scalar> b = p.backward({6,8,14,16});

    vector<uint32_t> seeds(100);
    std::seed_seq ss{8, 6, 7, 5, 3, 0, 9}; // TODO: change this from hard-coded?
    ss.generate(seeds.begin(), seeds.end());

    scalar lr = 0.1;

    std::uniform_real_distribution<double> unif(-.5, .5);

    mt19937_64 rng(seeds[0]);

    vector<vector<vector<scalar>>> kers(3, vector<vector<scalar>>(5, vector<scalar>(5)));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                kers[i][j][k] = unif(rng); 
            }
        }
    }

    mt19937_64 rng_2(seeds[1]);
    vector<vector<vector<scalar>>> kers_2(3, vector<vector<scalar>>(6, vector<scalar>(6)));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 6; k++) {
                kers_2[i][j][k] = unif(rng_2); 
            }
        }
    }

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(300, 200, 3, 0, kers, lr)));
    n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer(300 - 5 + 1, 200 - 5 + 1, 3, 2)));
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer((300 -5 + 1)/2, (200 -5 + 1)/2, 3, 0, kers_2, lr)));
    n.add(make_shared<MaxPoolingLayer>(MaxPoolingLayer((300 -5 + 1)/2 - 6 + 1, (200 - 5 + 1)/2 - 6 + 1, 3, 2)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(143 * 93, 2000, lr, seeds[2])));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(2000, hyp_tan, hyp_tan_der)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(2000, 90, lr, seeds[3])));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(90, hyp_tan, hyp_tan_der)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(90, 9, lr, seeds[4])));
    n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(9, hyp_tan, hyp_tan_der)));
    
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
    // plt::save("plots/animals_training.pdf");

    // n.save_weights("data/iris_weights.txt");
    return 0;
}