#include <iostream>
#include <NeuralNetwork.h>
#include <Python.h>
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

int main()
{
  vector<uint32_t> seeds(100);
  std::seed_seq ss{8, 6, 7, 5, 3, 0, 9}; // TODO: change this from hard-coded?
  ss.generate(seeds.begin(), seeds.end());

  scalar lr = 0.1;

  cout << "Hello world" << endl;
  NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
  n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(2, 3, lr, seeds[0])));
  n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(3, hyp_tan, hyp_tan_der)));
  n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(3, 1, lr, seeds[1])));
  n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(1, hyp_tan, hyp_tan_der)));

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

  vector<scalar> epoch_data = n.train(X, Y, 1000);

  vector<vector<scalar>> output;
  for (size_t i = 0; i < X.size(); i++) {
      output.push_back(n.predict(X[i]));
  }

  vector<int> epochs(1000);
  iota(epochs.begin(), epochs.end(), 1);

  plt::plot(epochs, epoch_data);
  plt::xlabel("Epoch");
  plt::ylabel("Mean Squared Error");
  plt::title("XOR training");
  plt::save("plots/x_or_training.pdf");

  n.save_weights("data/weights.txt");

  // Py_Initialize();
  // PyRun_SimpleString();
  // Py_Finalize();

  return 0;
}