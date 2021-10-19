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
typedef unsigned int uint;
using namespace std;

class Layer {
public:
    Layer();
    Layer(uint n_in, uint n_out) : n_inputs(n_in), n_outputs(n_out) {
        if(!are_generated) {
            ss.generate(Layer::seeds.begin(), Layer::seeds.end());
            are_generated = true;
        }
    }

    virtual vector<scalar> forward(vector<scalar> input) = 0;   
    virtual vector<scalar> backward(vector<scalar> output_error) = 0;
    vector<scalar> in;

    char get_layer_type();
    inline uint get_n_inputs() {return n_inputs;};
    inline uint get_n_outputs() {return n_outputs;};
protected:
    uint n_inputs;
    uint n_outputs;

    static bool are_generated; // this is a bad way to initalize the static seeds
    // but they need to be generated with ss.generate() which is not 
    // a nice constant expression
    static std::seed_seq ss;
    static uint num_seeds_used;
    static vector<uint32_t> seeds;
    static std::mt19937_64 rng;
};

#endif