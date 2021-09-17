#pragma once
#ifndef CONVOLUTIONLAYER_H_
#define CONVOLUTIONLAYER_H_

#include <Layer.h>

// TODO: Weight initialization

class ConvolutionLayer : public Layer {
public:
    /* I would include a stride length, but for simplicity use length 1
     * This shouldn't be too bad as there is no loss of info here
     *
     * Likewise, for simplicity use kernels that are square 
     * 
     * kernels.size() == num_channels
     * 
     * Throughout layer, work with a flattened array partioned into num_channels chunks
     */
    ConvolutionLayer(int width,
                     int height,
                     int num_channels,
                     int padding_size,
                     int kernel_size,
                     scalar learning_rate);

    vector<scalar> forward(vector<scalar> input);
    vector<scalar> backward(vector<scalar> output_error);

    inline vector<vector<vector<scalar>>> get_kernels() {return kernels;}
protected:
    int height;
    int width;

    int num_channels;
    int padding_size; 
    int kernel_size;

    vector<vector<vector<scalar>>> kernels;

    scalar learning_rate;

    int input_index(int vert, int horiz, int channel);
    int output_index(int vert, int horiz, int channel);
    scalar padded_value(int vert, int horiz, int channel);
    bool is_padding(int vert, int horiz);

    int output_height;
    int output_width;

    void initialize_kernels();

};

#endif