#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "bevdet.h"
#include "utils.h"
#include "cpu_jpegdecoder.h"
#include <yaml-cpp/yaml.h>

void TestNuscenes(YAML::Node &config)
{
    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
    std::string data_info_path = config["dataset_info"].as<std::string>();
    std::string model_config = config["ModelConfig"].as<std::string>();
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    std::string output_dir = config["OutputDir"].as<std::string>();
    std::vector<std::string> cams_name =
        config["cams"].as<std::vector<std::string>>();

    DataLoader nuscenes(img_N, img_h, img_w, data_info_path, cams_name);
    BEVDet bevdet(model_config, img_N, nuscenes.get_cams_intrin(),
                  nuscenes.get_cams2ego_rot(), nuscenes.get_cams2ego_trans(),
                  imgstage_file, bevstage_file);
    std::vector<Box> ego_boxes;
    double sum_time = 0;
    int cnt = 0;
    for (int i = 0; i < nuscenes.size(); i++)
    {
        ego_boxes.clear();
        float time = 0.f;
        bevdet.DoInfer(nuscenes.data(i), ego_boxes, time, i);
        if (i != 0)
        {
            sum_time += time;
            cnt++;
        }
        Boxes2Txt(ego_boxes,
                  output_dir + "/bevdet_egoboxes_" + std::to_string(i) + ".txt",
                  true);
    }
    printf("Infer mean cost time : %.5lf ms\n", sum_time / cnt);
}

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
        imgs_file.push_back(file.second.as<std::string>()); //../sample0/imgs/CAM_FRONT_LEFT.jpg
        cam_names.push_back(file.first.as<std::string>());  // CAM_FRONT_LEFT
    }

    camsData sampleData;
    sampleData.param = camParams(camconfig, img_N, cam_names);

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
    bool testNuscenes = config["TestNuscenes"].as<bool>();
    if (testNuscenes)
        TestNuscenes(config);

    else
        test_infer(config);

    return 0;
}