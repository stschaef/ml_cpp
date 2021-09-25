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
    uint total_epochs = 2;
    uint batch_size = 2048;
    uint num_channels = 3;
    scalar lr = 0.1;

    NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(300, 200, num_channels, 0, 11, lr)));
    n.add(make_shared<ReLULayer>(ReLULayer(290 * 190 * num_channels))); 
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(291, 191, num_channels, 0, 16, lr)));
    n.add(make_shared<TanhLayer>(TanhLayer(275 * 175 * num_channels)));
    n.add(make_shared<AveragePoolingLayer>(AveragePoolingLayer(275, 175, num_channels, 2)));
    n.add(make_shared<ConvolutionLayer>(ConvolutionLayer(55, 35, num_channels, 0, 26, lr)));
    n.add(make_shared<ReLULayer>(ReLULayer(30 * 10 * num_channels))); 
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(30 * 10 * num_channels, 128 , lr)));
    n.add(make_shared<TanhLayer>(TanhLayer(128)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(128, 64 , lr)));
    n.add(make_shared<TanhLayer>(TanhLayer(64)));
    n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(64, 9, lr)));
    
    vector<vector<scalar>> X_test, Y_test, X_train, Y_train;


    vector<string> animal_filepaths = animal_images();
    random_shuffle(animal_filepaths.begin(), animal_filepaths.end());
    
    for (int i = 0; i < int(animal_filepaths.size() * .7) ; i++) {
        vector<scalar> label(9, 0.0);
        string category = split(animal_filepaths[i], '/')[6];
        label[animals_category_to_index(category)] = 1.0;

        X_train.push_back(flatten_animals_image(animal_filepaths[i]));
        Y_train.push_back(label);
    }

    for (int i = int(animal_filepaths.size() * .7); i < int(animal_filepaths.size()) ; i++) {
        vector<scalar> label(9, 0);
        string category = split(animal_filepaths[i], '/')[6];
        label[animals_category_to_index(category)] = 1;

        X_test.push_back(flatten_animals_image(animal_filepaths[i]));
        Y_test.push_back(label);
    }
    
    vector<scalar> testing_accuracy;
    vector<scalar> epoch_data = n.train(X_train,
                                        Y_train,
                                        total_epochs,
                                        batch_size,
                                        X_test,
                                        Y_test,
                                        testing_accuracy);

    n.save_weights("/home/stschaef/ml_cpp/data/animal_weights.txt");

    vector<int> epochs(total_epochs);
    iota(epochs.begin(), epochs.end(), 1);

    plt::plot(epochs, testing_accuracy);
    plt::xlabel("Number of Epochs");
    plt::ylabel("Test Set Accuracy/MSE");

    plt::plot(epochs, epoch_data);
    
    plt::title("Animals: Test Accuracy and Training Loss (MSE)");
    // plt::show();
    plt::save("/home/stschaef/ml_cpp/plots/animals_training.pdf");


    return 0;
}