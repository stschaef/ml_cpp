#include <iostream>
#include <NeuralNetwork.h>
#include <Python.h>
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

u_char** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

int main()
{
  vector<uint32_t> seeds(100);
  std::seed_seq ss{8, 6, 7, 5, 3, 0, 9}; // TODO: change this from hard-coded?
  ss.generate(seeds.begin(), seeds.end());

  scalar lr = 0.1;

//   cout << "Hello world" << endl;
//   NeuralNetwork n(lr, mean_squared_error, mean_squared_error_der);
//   n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(2, 3, lr, seeds[0])));
//   n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(3, hyp_tan, hyp_tan_der)));
//   n.add(make_shared<FullyConnectedLayer>(FullyConnectedLayer(3, 1, lr, seeds[1])));
//   n.add(make_shared<ActivationFunctionLayer>(ActivationFunctionLayer(1, hyp_tan, hyp_tan_der)));

//   vector<vector<scalar>> X;
//   vector<vector<scalar>> Y;
//   vector<scalar> epoch_data = n.train(X, Y, 1000, 32);

//   vector<vector<scalar>> output;
//   for (size_t i = 0; i < X.size(); i++) {
//       output.push_back(n.predict(X[i]));
//   }

//   vector<int> epochs(1000);
//   iota(epochs.begin(), epochs.end(), 1);

//   plt::plot(epochs, epoch_data);
//   plt::xlabel("Number of Epochs");
//   plt::ylabel("Mean Squared Error");
//   plt::title("MNIST training");
//   plt::save("plots/mnist_no_cnn_training.pdf");

//   n.save_weights("data/mnist_no_cnnweights.txt");

    u_char** a = read_mnist_images("data/t10k-images-idx3-ubytes", 1, 28*28);

  return 0;
}