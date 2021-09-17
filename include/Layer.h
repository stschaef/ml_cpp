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
    Layer(int n_in, int n_out) : n_inputs(n_in), n_outputs(n_out) {
        if(!are_generated) {
            ss.generate(Layer::seeds.begin(), Layer::seeds.end());
            are_generated = true;
        }
    }

    virtual vector<scalar> forward(vector<scalar> input) = 0;   
    virtual vector<scalar> backward(vector<scalar> output_error) = 0;
    vector<scalar> in;

    char get_layer_type();
    inline int get_n_inputs() {return n_inputs;};
    inline int get_n_outputs() {return n_outputs;};
protected:
    int n_inputs;
    int n_outputs;

    static bool are_generated; // this is a bad way to initalize the static seeds
    // but they need to be generated with ss.generate() which is not 
    // a nice constant expression
    static std::seed_seq ss;
    static int num_seeds_used;
    static vector<int32_t> seeds;
    static std::mt19937_64 rng;
};

#endif