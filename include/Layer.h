#pragma once
#ifndef LAYER_H_
#define LAYER_H_


#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>

typedef float scalar;
using namespace std;

class Layer {
public:
    Layer();
    Layer(uint n_in, uint n_out) : n_inputs(n_in), n_outputs(n_out) {}

    virtual vector<scalar> forward(vector<scalar> input) = 0;   
    virtual vector<scalar> backward(vector<scalar> output_error) = 0;
    vector<scalar> in;

    char get_layer_type();
    inline uint get_n_inputs() {return n_inputs;};
    inline uint get_n_outputs() {return n_outputs;};
protected:
    uint n_inputs;
    uint n_outputs;
};

#endif