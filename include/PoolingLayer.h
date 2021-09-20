#pragma once
#ifndef POOLINGLAYER_H_
#define POOLINGLAYER_H_

#include <ConvolutionLayer.h>

class PoolingLayer : public ConvolutionLayer {
public:
    PoolingLayer(uint width,
                 uint height,
                 uint num_channels,
                 uint pooling_size);
    
    virtual vector<scalar> forward(vector<scalar> input) = 0;
    virtual vector<scalar> backward(vector<scalar> output_error) = 0;
protected:
    uint pooling_size;
};

class MaxPoolingLayer : public PoolingLayer {
public:
    MaxPoolingLayer(uint width,
                    uint height,
                    uint num_channels,
                    uint pooling_size);
    
    vector<scalar> forward(vector<scalar> input);
    vector<scalar> backward(vector<scalar> output_error);
private:
    vector<uint> max_homes;
};

class AveragePoolingLayer : public PoolingLayer {
public:
    AveragePoolingLayer(uint height,
                        uint width,
                        uint num_channels,
                        uint pooling_size);
    
    vector<scalar> forward(vector<scalar> input);
    vector<scalar> backward(vector<scalar> output_error);
};

#endif