#include <iostream>
#include <NeuralNetwork.h>
#include <Python.h>
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

int main()
{
  scalar lr = 0.1;

  NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);

  // n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(2, 3, lr)));
  // n.add(make_shared<TanhLayer>(TanhLayer(3)));
  // n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(3, 1, lr)));
  // n.add(make_shared<TanhLayer>(TanhLayer(1)));

  vector<vector<scalar>> X;
  X.push_back(vector<scalar>({0 , 0}));
  X.push_back(vector<scalar>({0 , 1}));
  X.push_back(vector<scalar>({1 , 0}));
  X.push_back(vector<scalar>({1 , 1}));

  vector<vector<scalar>> Y;
  Y.push_back(vector<scalar>({0}));
  Y.push_back(vector<scalar>({1}));
  Y.push_back(vector<scalar>({1}));
  Y.push_back(vector<scalar>({0}));

  n.load_weights("/home/stschaef/ml_cpp/data/xor_weights.txt");

  vector<scalar> testing_acc;
  vector<scalar> epoch_data = n.train(X, Y, 10, 1, X, Y, testing_acc);

  vector<int> epochs(10);
  iota(epochs.begin(), epochs.end(), 1);

  // plt::plot(epochs, epoch_data);
  // plt::xlabel("Number of Iterations");
  // plt::ylabel("Mean Squared Error");
  // plt::title("XOR training");
  // plt::save("plots/x_or_training.pdf");
  // plt::show();

  // n.save_weights("/home/stschaef/ml_cpp/data/xor_weights.txt");
  return 0;
}