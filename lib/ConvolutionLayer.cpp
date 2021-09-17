#include <ConvolutionLayer.h>

// ___________ConvolutionLayer Implementations___________

ConvolutionLayer::ConvolutionLayer(
    int height,
    int width,
    int num_channels,
    int padding_size,
    int kernel_size,
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

int ConvolutionLayer::input_index(int vert, int horiz, int channel) 
{   
    int v = vert - padding_size;
    int h = horiz - padding_size;
    return channel * height * width + width * v  + h;
}

int ConvolutionLayer::output_index(int vert, int horiz, int channel) 
{   
    return channel * output_height * output_width + output_width * vert + horiz;
}

scalar ConvolutionLayer::padded_value(int vert, int horiz, int channel)
{
    return 0 * horiz * vert * channel;
}

bool ConvolutionLayer::is_padding(int vert, int horiz)
{
    return vert < padding_size || vert >= padding_size + height || \
           horiz < padding_size || horiz >= padding_size + width;
}

vector<scalar> ConvolutionLayer::forward(vector<scalar> input)
{
    in = input;
    vector<scalar> output(n_outputs);

    for (int c = 0 ; c < num_channels; c++) {
        vector<vector<scalar>> kernel = kernels[c];
        for (int i = 0; i < height + 2 * padding_size - kernel_size + 1; i++) {
            // i == vertical coord of top left of kernel
            for (int j = 0; j < width + 2 * padding_size - kernel_size + 1; j++) {
                // j == horiz coord of top left of kernel

                scalar sum = 0;

                for (int y = i; y < i + kernel_size; y++) {
                    // y == absolute vertical coord
                    for (int x = j; x < j + kernel_size; x++) {
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

    for (int a = 0; a < kernel_size; a++) {
        for (int b = 0; b < kernel_size; b++){
            vector<scalar> sums(num_channels, scalar(0));

            for (int i = 0; i < height + 2 * padding_size - kernel_size + 1; i++) {
                for (int j = 0; j < width + 2 * padding_size - kernel_size + 1; j++) {
                    for (int c = 0; c < num_channels; c++) {
                        sums[c] += output_error[output_index(i, j, c)] * in[input_index(a + i, b + j, c)];
                        input_error[input_index(a + i, b + j, c)] += output_error[output_index(i, j, c)] * kernels[c][a][b];
                    }
                }
            }

            for (int c = 0; c < num_channels; c++) {
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

    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j< kernel_size; j++) {
                kernels[c][i][j] = unif(rng);
            }
        }
    }
}