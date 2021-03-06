#pragma once
#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <Layer.h>
#include <ActivationFunctionLayer.h>
#include <FullyConnectedLayer.h>
#include <ConvolutionLayer.h>
#include <PoolingLayer.h>
#include <sstream>

class NeuralNetwork {
public:
    NeuralNetwork(scalar learning_rate,
                  function<scalar(vector<scalar>, vector<scalar>)> loss,
                  function<vector<scalar>(vector<scalar>, vector<scalar>)> loss_der);

    vector<scalar> predict(vector<scalar> input);

    vector<scalar> train(vector<vector<scalar>>& input_data,
               vector<vector<scalar>>& actual,
               uint num_epochs,
               uint batch_size,
               vector<vector<scalar>>& X_test,
               vector<vector<scalar>>& Y_test,
               vector<scalar>& testing_accuracy);

    scalar test(vector<vector<scalar>>& X_test,
                vector<vector<scalar>>& Y_test);

    void add(shared_ptr<Layer> l);

    void save_weights(string out_filename);
    void load_weights(string in_filename);
private:
    vector<shared_ptr<Layer>> layers;

    void generate_seeds();
    vector<uint32_t> seeds;

    scalar learning_rate;

    function<scalar(vector<scalar>, vector<scalar>)> loss;
    function<vector<scalar>(vector<scalar>, vector<scalar>)> loss_der;
};

// ___________Loss Functions___________
inline scalar mean_squared_error(vector<scalar> actual, vector<scalar> predicted)
{
    scalar sum = 0;
    int n = actual.size();
    for (int i = 0; i < n; i++) {
        sum += pow(actual[i] - predicted[i], 2);
    }
    sum *= 1.0 / n;
    return sum;
}

inline vector<scalar> mean_squared_error_der(vector<scalar> actual, vector<scalar> predicted)
{   
    int n = actual.size();
    for (int i = 0; i < n; i++) {
        actual[i] = predicted[i] - actual[i];
        actual[i] *= 2.0 / n;
    }

    return actual;
}

#endif