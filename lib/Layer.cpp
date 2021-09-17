#include <Layer.h>
#include <ActivationFunctionLayer.h>
#include <FullyConnectedLayer.h>
#include <ConvolutionLayer.h>
#include <PoolingLayer.h>

// ___________Layer Implementations___________
char Layer::get_layer_type()
{
    // Use this for storing weight information into files
    // Encode each layer type with a char
    if (dynamic_cast<FullyConnectedLayer*>(this)) return 'a';
    if (dynamic_cast<ActivationFunctionLayer*>(this)) return 'b';
    if (dynamic_cast<MaxPoolingLayer*>(this)) return 'c';
    if (dynamic_cast<AveragePoolingLayer*>(this)) return 'd';
    if (dynamic_cast<ConvolutionLayer*>(this)) return 'e';
    return -1;
}

Layer::Layer() {
    if(!are_generated) {
        ss.generate(Layer::seeds.begin(), Layer::seeds.end());
        are_generated = true;
    }
}

bool Layer::are_generated = false;
int Layer::num_seeds_used = 0;
std::seed_seq Layer::ss{8, 6, 7, 5, 3, 0, 9};
vector<int32_t> Layer::seeds(100);
std::mt19937_64 Layer::rng(0);

