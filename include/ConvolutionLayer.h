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
    ConvolutionLayer(uint width,
                     uint height,
                     uint num_channels,
                     uint padding_size,
                     uint kernel_size,
                     scalar learning_rate);

    vector<scalar> forward(vector<scalar> input);
    vector<scalar> backward(vector<scalar> output_error);

    void set_kernel_at(uint channel_num, uint i, uint j, scalar val);

    inline vector<vector<vector<scalar>>> get_kernels() {return kernels;}
    inline uint get_height() {return height;}
    inline uint get_width() {return width;}
    inline uint get_num_channels() {return num_channels;}
    inline uint get_padding_size() {return padding_size;}
    inline uint get_kernel_size() {return kernel_size;}

protected:
    uint height;
    uint width;

    uint num_channels;
    uint padding_size; 
    uint kernel_size;

    vector<vector<vector<scalar>>> kernels;

    scalar learning_rate;

    uint input_index(uint vert, uint horiz, uint channel);
    uint output_index(uint vert, uint horiz, uint channel);
    scalar padded_value(uint vert, uint horiz, uint channel);
    bool is_padding(uint vert, uint horiz);

    uint output_height;
    uint output_width;

    void initialize_kernels();

};

#endif