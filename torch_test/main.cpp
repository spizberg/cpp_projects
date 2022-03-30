#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <array>
#include <chrono>

const std::array<std::string, 15> class_names { "AQS100 D",
                                                "AQS120 D",
                                                "AQUADOTS D",
                                                "AQUAFUN D",
                                                "BASKET LERIA E",
                                                "BASKET LUKA E",
                                                "BOOTS KURKUMIN E",
                                                "BOOTS MARIBEL E",
                                                "MOCASSIN ALESSIO E",
                                                "MULE VALAMA E",
                                                "NH100 D",
                                                "SAILING100 D",
                                                "SANDALE JAIDA E",
                                                "SANDALE PENSEE E19 E",
                                                "TS100 D" };

at::Tensor get_tensor(const std::string &filename){
    cv::Mat img {cv::imread(filename)};
    cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    cv::Mat matFloat;
    img.convertTo(matFloat, CV_32F, 1.0 / 255);
    auto tensor = torch::from_blob(matFloat.data, {1, 224, 224, 3});
    tensor = tensor.permute({0, 3, 1, 2});
    std::vector<double> norm_mean = {0.485, 0.456, 0.406};
    std::vector<double> norm_std = {0.229, 0.224, 0.225};
    tensor = torch::data::transforms::Normalize(norm_mean, norm_std)(tensor);
    return tensor;
}

torch::jit::script::Module get_model(const std::string &model_path){
    torch::jit::script::Module module;
    module = torch::jit::load(model_path);
    return module;
}

int main(int argc, char** argv){
    if (argc < 2)
        return -1;

    const std::string filename{argv[1]};
    const std::string model_path{"/home/nathan/Code/notebooks_folder/libtorch_models/traced_idshoes_resnet_model_int8.pt"};

    const at::Tensor input_tensor{get_tensor(filename)};

    torch::jit::script::Module model{get_model(model_path)};
    model.eval();

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Execute the model and turn its output into a tensor.
    auto start = std::chrono::high_resolution_clock::now();
    const at::Tensor output = model.forward(inputs).toTensor();
    auto stop = std::chrono::high_resolution_clock::now();
    const int index {output.argmax().item().to<int>()};
    std::cout << class_names[index] << '\n';
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "\n";
    return 0;
}
