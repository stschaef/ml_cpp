#pragma once
#ifndef UTILS_H_
#define UTILS_H_

#include <tuple>
#include <Layer.h>
#include <opencv4/opencv2/opencv.hpp>  

vector<scalar> flatten_animals_image(string image_path);
vector<scalar> flatten_mnist_image(string image_path, uint num_channels);

#endif