#include <utils.h>

vector<scalar> flatten_animals_image(string image_path)
{
    cv::Mat image;
    image = cv::imread(image_path, cv::IMREAD_COLOR);

    vector<scalar> reds;
    vector<scalar> blues;
    vector<scalar> greens;
    for (int i = 0; i < image.rows; i++) {
        cv::Vec3b* pixel = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image.cols; j++) {
            reds.push_back(scalar(pixel[j][2]) / 255.0);
            blues.push_back(scalar(pixel[j][1]) / 255.0);
            greens.push_back(scalar(pixel[j][0]) / 255.0);
        }
    }

    vector<scalar> output = reds;
    output.reserve(3 * reds.size());
    output.insert(output.end(), blues.begin(), blues.end());
    output.insert(output.end(), greens.begin(), greens.end());

    return output;
}

vector<scalar> flatten_mnist_image(string image_path, uint num_channels)
{
    cv::Mat image;
    image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    vector<scalar> img_vec;
    img_vec.reserve(image.rows * image.cols);
    for(int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            img_vec.push_back(image.at<uchar>(i, j) / 255.0); // normalize
        }
    }
    for (auto a: img_vec) {
        cout << a << '\n';
    }
    vector<scalar> output;
    output.reserve(num_channels * img_vec.size());
    for (uint i = 0; i < num_channels; i ++) {
        output.insert(output.end(), img_vec.begin(), img_vec.end());
    }
    
    return output;
}

vector<string> mnist_test(int digit)
{
    string test_folder = "/home/stschaef/ml_cpp/data/mnist/testing/" + to_string(digit) + "/*";
    vector<cv::String> filenames;
    cv::glob(test_folder, filenames, true);

    return filenames;
}

vector<string> mnist_train(int digit)
{
    string test_folder = "/home/stschaef/ml_cpp/data/mnist/training/" + to_string(digit) + "/*";;
    vector<cv::String> filenames;
    cv::glob(test_folder, filenames, true);

    return filenames;
}

vector<string> animal_images()
{
    string test_folder = "/home/stschaef/ml_cpp/data/animals_resized";
    vector<cv::String> filenames;
    cv::glob(test_folder, filenames, true);

    return filenames;
}

vector<string> split(const string &s, char delim) {
  stringstream ss(s);
  string item;
  vector<string> elems;
  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

int animals_category_to_index(string s) {
    // Used for encoding categorical variables
    // ex) butterfly corresponds to label [1, 0, 0, 0, ...]
    //     the 1 is in index animals_category_to_index("butterfly")
    if (s == "butterfly") {
        return 0;
    }
    if (s == "cat") {
        return 1;
    }
    if (s == "chicken") {
        return 2;
    }
    if (s == "cow") {
        return 3;
    }
    if (s == "dog") {
        return 4;
    }
    if (s == "elephant") {
        return 5;
    }
    if (s == "horse") {
        return 6;
    }
    if (s == "sheep") {
        return 7;
    }
    if (s == "squirrel") {
        return 8;
    }
    return -1;
}