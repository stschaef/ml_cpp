# Machine Learning in C++

This was a personal project to implement neural networks from scratch in C++. I was able to make the following 
1. fully connected layers 
2. convolution layers
3. pooling layers (max and average pooling)
4. activation function layers

This is enough to create a minimum viable convolutional neural net for computer vision. This is demonstrated by training on the MNIST handwritten digits dataset and achieving 97% accuracy on test data.

I am hoping to incorporate the digit-recognition model into a React app via WebAssembly. 

## Some Things I Would Change Given Enough Time
1. CUDA support for performance. CUDA does not play nicely with the STL or object-oriented design patterns, so including CUDA support would take a massive overhaul that I don't have the budget for timewise. 

2. Padding in the convolutional layers was never fully implemented. This was not much of a hindrance since my laptop couldn't handle large amounts of data or complicated network architectures.

3. Do proper serialization via something like Boost. To avoid the library dependencies, which are difficult when compiling via `emcc`, I homebaked model serialization into `.txt`'s. Storing with strings is clearly not an efficient use of storage, but is fine at the current scale.

4. More universal preprocessor guards. In particular, to better handle the use of `g++` vs `emcc`.

## Emscripten

Note that the target `predictor` is compiled with `emcc` rather than `g++`. Moreover, it is the only target meant to be used in this way.

This could've been accomplished with `emmake`, but that was throwing some weird errors and this worked for whatever reason.
