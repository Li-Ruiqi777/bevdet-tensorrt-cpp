#pragma once
#include <chrono>
#include <string>
#include <vector>
#include "include/bevdet.h"
#include "opencv2/opencv.hpp"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void Getinfo();
void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name,bool with_vel);

void Egobox2Lidarbox(const std::vector<Box> &ego_boxes,
                     std::vector<Box> &lidar_boxes,
                     const Eigen::Quaternion<float> &lidar2ego_rot,
                     const Eigen::Translation3f &lidar2ego_trans);

int cv2rawData(cv::Mat img, std::vector<char> &raw_data);
int cv2rawData(std::vector<cv::Mat> &imgs, std::vector<std::vector<char>> &raw_datas);