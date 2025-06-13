#include <iostream>
#include <yaml-cpp/yaml.h>
#include "common.h"
#include "bevdet.h"
#include "cpu_jpegdecoder.h"

void TestSample(YAML::Node &config)
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
    std::vector<std::string> imgs_name;
    for (auto file : sample)
    {
        imgs_file.push_back(file.second.as<std::string>());
        imgs_name.push_back(file.first.as<std::string>());
    }

    camsData sampleData;
    sampleData.param = camParams(camconfig, img_N, imgs_name);

    BEVDet bevdet(model_config, img_N, sampleData.param.cams_intrin,
                  sampleData.param.cams2ego_rot, sampleData.param.cams2ego_trans,
                  imgstage_file, bevstage_file);
    std::vector<std::vector<char>> imgs_data;
    read_sample(imgs_file, imgs_data);

    uchar *imgs_dev = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&imgs_dev,
                          img_N * 3 * img_w * img_h * sizeof(uchar)));
    decode_cpu(imgs_data, imgs_dev, img_w, img_h);
    sampleData.imgs_dev = imgs_dev;

    std::vector<Box> ego_boxes;
    ego_boxes.clear();
    float time = 0.f;
    bevdet.DoInfer(sampleData, ego_boxes, time);
    std::vector<Box> lidar_boxes;
    Egobox2Lidarbox(ego_boxes, lidar_boxes, sampleData.param.lidar2ego_rot,
                    sampleData.param.lidar2ego_trans);
    Boxes2Txt(lidar_boxes, output_lidarbox, false);
    ego_boxes.clear();
    bevdet.DoInfer(sampleData, ego_boxes, time); // only for inference time
}

int main()
{
    Getinfo();

    std::string config_file("../configure.yaml");
    YAML::Node config = YAML::LoadFile(config_file);
    printf("Successful load config : %s!\n", config_file.c_str());
  
    TestSample(config);

}