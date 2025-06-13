#pragma once
#include <chrono>
#include <string>
#include <vector>
#include "include/bevdet.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void Getinfo();
void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name,bool with_vel);

void Egobox2Lidarbox(const std::vector<Box> &ego_boxes,
                     std::vector<Box> &lidar_boxes,
                     const Eigen::Quaternion<float> &lidar2ego_rot,
                     const Eigen::Translation3f &lidar2ego_trans);