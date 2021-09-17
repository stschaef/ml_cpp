#pragma once
#ifndef UTILS_H_
#define UTILS_H_

#include <tuple>
#include <Layer.h>
#include <opencv4/opencv2/opencv.hpp>  
#include <glob.h>

vector<scalar> flatten_animals_image(string image_path);
vector<scalar> flatten_mnist_image(string image_path, uint num_channels);
vector<string> mnist_test(int digit);
vector<string> mnist_train(int digit);

#endif