#include <ConvolutionLayer.h>

// ___________ConvolutionLayer Implementations___________

ConvolutionLayer::ConvolutionLayer(
    uint height,
    uint width,
    uint num_channels,
    uint padding_size,
    uint kernel_size,
    scalar learning_rate) 
    : height(height),
      width(width),
      num_channels(num_channels),
      padding_size(padding_size),
      kernel_size(kernel_size),
      learning_rate(learning_rate)
{
    output_height = height + 2 * padding_size - kernel_size + 1;
    output_width = width + 2 * padding_size - kernel_size + 1;

    n_inputs = height * width * num_channels;
    n_outputs = output_height * output_width * num_channels;

    kernels = vector<vector<vector<scalar>>>(num_channels, 
        vector<vector<scalar>>(kernel_size, vector<scalar>(kernel_size, 0)));

    if (kernel_size > 0) initialize_kernels();
}

uint ConvolutionLayer::input_index(uint vert, uint horiz, uint channel) 
{   
    uint v = vert - padding_size;
    uint h = horiz - padding_size;
    return channel * height * width + width * v  + h;
}

uint ConvolutionLayer::output_index(uint vert, uint horiz, uint channel) 
{   
    return channel * output_height * output_width + output_width * vert + horiz;
}

scalar ConvolutionLayer::padded_value(uint vert, uint horiz, uint channel)
{
    return 0 * horiz * vert * channel;
}

bool ConvolutionLayer::is_padding(uint vert, uint horiz)
{
    return vert < padding_size || vert >= padding_size + height || \
           horiz < padding_size || horiz >= padding_size + width;
}

vector<scalar> ConvolutionLayer::forward(vector<scalar> input)
{
    in = input;
    vector<scalar> output(n_outputs);

    for (uint c = 0 ; c < num_channels; c++) {
        vector<vector<scalar>> kernel = kernels[c];
        for (uint i = 0; i < height + 2 * padding_size - kernel_size + 1; i++) {
            // i == vertical coord of top left of kernel
            for (uint j = 0; j < width + 2 * padding_size - kernel_size + 1; j++) {
                // j == horiz coord of top left of kernel

                scalar sum = 0;

                for (uint y = i; y < i + kernel_size; y++) {
                    // y == absolute vertical coord
                    for (uint x = j; x < j + kernel_size; x++) {
                        // x == absolute horiz coord
                        scalar value;
                        if (is_padding(y, x)) value = padded_value(y, x, c);
                        else value = input[input_index(y, x, c)];

                        sum += value * kernel[y - i][x - j];
                    }
                }

                output[output_index(i, j, c)] = sum;
            }   
        }
    }

    return output;
}

vector<scalar> ConvolutionLayer::backward(vector<scalar> output_error)
{
    // Same as FullyConnectedLayer
    vector<scalar> input_error(n_inputs, 0);

    for (uint a = 0; a < kernel_size; a++) {
        for (uint b = 0; b < kernel_size; b++){
            vector<scalar> sums(num_channels, scalar(0));

            for (uint i = 0; i < height + 2 * padding_size - kernel_size + 1; i++) {
                for (uint j = 0; j < width + 2 * padding_size - kernel_size + 1; j++) {
                    for (uint c = 0; c < num_channels; c++) {
                        sums[c] += output_error[output_index(i, j, c)] * in[input_index(a + i, b + j, c)];
                        input_error[input_index(a + i, b + j, c)] += output_error[output_index(i, j, c)] * kernels[c][a][b];
                    }
                }
            }

            for (uint c = 0; c < num_channels; c++) {
                kernels[c][a][b] -= learning_rate * sums[c];
            }
        }
    }
    return input_error;
}

void ConvolutionLayer::initialize_kernels()
{
    rng.seed(seeds[num_seeds_used++]);

    std::uniform_real_distribution<scalar> unif(-.5, .5);

    for (uint c = 0; c < num_channels; c++) {
        for (uint i = 0; i < kernel_size; i++) {
            for (uint j = 0; j< kernel_size; j++) {
                kernels[c][i][j] = unif(rng);
            }
        }
    }
}

void ConvolutionLayer::set_kernel_at(uint channel_num, uint i, uint j, scalar val)
{
    kernels[channel_num][i][j] = val;
}