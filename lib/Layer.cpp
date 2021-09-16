#include <Layer.h>
#include <ActivationFunctionLayer.h>
#include <FullyConnectedLayer.h>
#include <ConvolutionLayer.h>

// ___________Layer Implementations___________
char Layer::get_layer_type()
{
    // Use this for storing weight information into files
    // Encode each layer type with a char
    if (dynamic_cast<FullyConnectedLayer*>(this)) return 'f';
    if (dynamic_cast<ActivationFunctionLayer*>(this)) return 'a';
    if (dynamic_cast<ConvolutionLayer*>(this)) return 'c';
    return 'x';
}

Layer::Layer() {}
