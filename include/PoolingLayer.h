#pragma once
#ifndef POOLINGLAYER_H_
#define POOLINGLAYER_H_

#include <ConvolutionLayer.h>

class PoolingLayer : public ConvolutionLayer {
public:
    PoolingLayer(int height,
                 int width,
                 int num_channels,
                 int pooling_size);
    
    virtual vector<scalar> forward(vector<scalar> input) = 0;
    virtual vector<scalar> backward(vector<scalar> output_error) = 0;
protected:
    int pooling_size;
};

class MaxPoolingLayer : public PoolingLayer {
public:
    MaxPoolingLayer(int height,
                    int width,
                    int num_channels,
                    int pooling_size);
    
    vector<scalar> forward(vector<scalar> input);
    vector<scalar> backward(vector<scalar> output_error);
private:
    vector<int> max_homes;
};

class AveragePoolingLayer : public PoolingLayer {
public:
    AveragePoolingLayer(int height,
                        int width,
                        int num_channels,
                        int pooling_size);
    
    vector<scalar> forward(vector<scalar> input);
    vector<scalar> backward(vector<scalar> output_error);
};

#endif