#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "opencv2/core/types.hpp"
#include "utils.h"
#include "bevdet.h"
#include "cpu_jpegdecoder.h"

#include "opencv2/opencv.hpp"

void test_infer(YAML::Node &config)
{
    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
    std::string model_config = config["ModelConfig"].as<std::string>();
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    YAML::Node camconfig = YAML::LoadFile(config["CamConfig"].as<std::string>());
    std::string output_lidarbox = config["OutputLidarBox"].as<std::string>();
    YAML::Node sample = config["sample"];

    std::vector<std::string> imgs_file;
    std::vector<std::string> cam_names;

    for (auto file : sample)
    {
        imgs_file.push_back(file.second.as<std::string>());
        cam_names.push_back(file.first.as<std::string>());
    }

    // 构造相机参数
    camsData sampleData;
    sampleData.param = camParams(camconfig, img_N, cam_names);

    // 构造BEVDet对象
    BEVDet bevdet(model_config, img_N, sampleData.param.cams_intrin,
                  sampleData.param.cams2ego_rot, sampleData.param.cams2ego_trans,
                  imgstage_file, bevstage_file);

    // 读取输入数据
    // std::vector<std::vector<char>> img_raw_datas; // 原始图像二进制数据(没decode)
    // read_sample(imgs_file, img_raw_datas);
    std::vector<cv::Mat> imgs;
    std::vector<std::vector<char>> img_raw_datas;
    
    imgs.emplace_back(cv::imread("../sample0/imgs/person2.jpg", cv::IMREAD_COLOR));
    for(auto& img : imgs)
        cv::resize(img, img, cv::Size(img_w, img_h));

    cv2rawData(imgs, img_raw_datas);

    uchar *imgs_dev = nullptr; // 存在vram中
    CHECK_CUDA(cudaMalloc((void **)&imgs_dev,
                          img_N * 3 * img_w * img_h * sizeof(uchar)));

    
    // 在CPU上以jpg格式解码图片，并转换为BGRCHW格式，存到vram里
    decode_cpu(img_raw_datas, imgs_dev, img_w, img_h);
    sampleData.imgs_dev = imgs_dev;

    // 准备推理
    std::vector<Box> ego_boxes;
    float time = 0.f;
    bevdet.DoInfer(sampleData, ego_boxes, time);

    // 结果后处理
    std::vector<Box> lidar_boxes;
    Egobox2Lidarbox(ego_boxes, lidar_boxes, sampleData.param.lidar2ego_rot,
                    sampleData.param.lidar2ego_trans);
    Boxes2Txt(lidar_boxes, output_lidarbox, false);
}

int main()
{
    Getinfo();

    std::string config_file("../test_configure.yaml");
    YAML::Node config = YAML::LoadFile(config_file);
    printf("Successful load config : %s!\n", config_file.c_str());
  
    test_infer(config);

}