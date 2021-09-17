// #include <utils.h>

// vector<scalar> flatten_animals_image(string image_path)
// {
//     cv::Mat image;
//     image = cv::imread(image_path, cv::IMREAD_COLOR);

//     vector<scalar> reds;
//     vector<scalar> blues;
//     vector<scalar> greens;
//     for (int i = 0; i < image.rows; i++) {
//         cv::Vec3b* pixel = image.ptr<cv::Vec3b>(i);
//         for (int j = 0; j < image.cols; j++) {
//             reds.push_back(scalar(pixel[j][2]));
//             blues.push_back(scalar(pixel[j][1]));
//             greens.push_back(scalar(pixel[j][0]));
//         }
//     }

//     vector<scalar> output = reds;
//     output.reserve(3 * reds.size());
//     output.insert(output.end(), blues.begin(), blues.end());
//     output.insert(output.end(), greens.begin(), greens.end());

//     return output;
// }

// vector<scalar> flatten_mnist_image(string image_path, int num_channels)
// {
//     cv::Mat image;
//     image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

//     vector<scalar> img_vec;
//     img_vec.reserve(image.rows * image.cols);
//     for(int i = 0; i < image.rows; i++) {
//         for (int j = 0; j < image.cols; j++) {
//             img_vec.push_back(image.at<uchar>(i, j) / 255.0); // normalize
//         }
//     }
//     vector<scalar> output;
//     output.reserve(num_channels * img_vec.size());
//     for (int i = 0; i < num_channels; i ++) {
//         output.insert(output.end(), img_vec.begin(), img_vec.end());
//     }
    
//     return output;
// }

// vector<string> mnist_test(int digit)
// {
//     string test_folder = "/home/stschaef/ml_cpp/data/mnist/testing/" + to_string(digit) + "/*";
//     vector<cv::String> filenames;
//     cv::glob(test_folder, filenames, true);

//     return filenames;
// }

// vector<string> mnist_train(int digit)
// {
//     string test_folder = "/home/stschaef/ml_cpp/data/mnist/training/" + to_string(digit) + "/*";;
//     vector<cv::String> filenames;
//     cv::glob(test_folder, filenames, true);

//     return filenames;
// }