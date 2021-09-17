#include <PoolingLayer.h>

// ___________PoolingLayer Implementations___________

PoolingLayer::PoolingLayer(
    int height,
    int width,
    int num_channels,
    int pooling_size)
    : ConvolutionLayer(height,
                       width,
                       num_channels,
                       0,
                       0,
                       0.0),
      pooling_size(pooling_size) {}

// ___________MaxPoolingLayer Implementations___________
MaxPoolingLayer::MaxPoolingLayer(
    int width,
    int height,
    int num_channels,
    int pooling_size)
    : PoolingLayer(height,
                   width,
                   num_channels,
                   pooling_size)
{
    n_outputs = (height / pooling_size) * (width / pooling_size) * num_channels;
}

vector<scalar> MaxPoolingLayer::forward(vector<scalar> input)
{
    in = input;
    vector<scalar> output;
    max_homes.clear();

    for (int c = 0 ; c < num_channels; c++) {
        vector<vector<scalar>> kernel = kernels[c];
        for (int i = 0; i < height; i = i + pooling_size) {
            for (int j = 0; j < width; j = j + pooling_size) {
                scalar biggest = scalar(std::numeric_limits<double>::lowest());
                int biggest_index = -1;

                for (int y = i; y < i + pooling_size; y++) {
                    for (int x = j; x < j + pooling_size; x++) {
                        scalar value;
                        if (is_padding(y, x)) value = padded_value(y, x, c);
                        else value = input[input_index(y, x, c)];

                        biggest_index = (value > biggest) ? input_index(y, x, c) : biggest_index;
                        biggest = (value > biggest) ? value : biggest;
                    }
                }

                output.push_back(biggest);
                max_homes.push_back(biggest_index);
            }   
        }
    }

    return output;
}

vector<scalar> MaxPoolingLayer::backward(vector<scalar> output_error)
{
    vector<scalar> input_error(n_inputs, 0);
    for (size_t i = 0; i < max_homes.size(); i++) {
        input_error[max_homes[i]] = output_error[i];
    }

    return input_error;
}

// ___________AveragePoolingLayer Implementations___________
AveragePoolingLayer::AveragePoolingLayer(
    int width,
    int height,
    int num_channels,
    int pooling_size)
    : PoolingLayer(height,
                   width,
                   num_channels,
                   pooling_size)
{
    n_outputs = (height / pooling_size) * (width / pooling_size) * num_channels;
}

vector<scalar> AveragePoolingLayer::forward(vector<scalar> input)
{
    in = input;
    vector<scalar> output;

    for (int c = 0 ; c < num_channels; c++) {
        vector<vector<scalar>> kernel = kernels[c];
        for (int i = 0; i < height; i = i + pooling_size) {
            for (int j = 0; j < width; j = j + pooling_size) {
                scalar sum = scalar(0);
                for (int y = i; y < i + pooling_size; y++) {
                    for (int x = j; x < j + pooling_size; x++) {
                        scalar value;
                        if (is_padding(y, x)) value = padded_value(y, x, c);
                        else value = input[input_index(y, x, c)];

                        sum += value;
                    }
                }
                output.push_back(sum / scalar(pooling_size * pooling_size));
            }   
        }
    }

    return output;
}

vector<scalar> AveragePoolingLayer::backward(vector<scalar> output_error)
{
    vector<scalar> output(n_inputs, 0);
    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int out_height = height / pooling_size;
                int out_width = width / pooling_size;
                int out_idx = c * out_height * out_width + (i / pooling_size) * out_width + (j / pooling_size);
                output[input_index(i, j, c)] = output_error[out_idx];
            }
        }
    }

    return output;
}