#ifndef __DATA_H__
#define __DATA_H__

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <yaml-cpp/yaml.h>

#include "common.h"
#include "nvjpegdecoder.h"

// 相机参数结构体
struct camParams
{
    camParams() = default;
    // 从YAML配置初始化相机参数
    camParams(const YAML::Node &config, int n, std::vector<std::string> &cams_name);

    int N_img; // 相机数量

    Eigen::Quaternion<float> ego2global_rot; // 自车到全局坐标系的旋转
    Eigen::Translation3f ego2global_trans;   // 自车到全局坐标系的平移

    Eigen::Quaternion<float> lidar2ego_rot; // LiDAR到自车坐标系的旋转
    Eigen::Translation3f lidar2ego_trans;   // LiDAR到自车坐标系的平移

    std::vector<Eigen::Matrix3f> cams_intrin;           // 相机内参矩阵
    std::vector<Eigen::Quaternion<float>> cams2ego_rot; // 相机到自车坐标系的旋转
    std::vector<Eigen::Translation3f> cams2ego_trans;   // 相机到自车坐标系的平移

    std::vector<std::string> imgs_file; // 图像文件路径

    unsigned long long timestamp; // 时间戳
    std::string scene_token;      // 场景标识符
};

// 相机数据结构体
struct camsData
{
    camsData() = default;
    camsData(const camParams &_param)
        : param(_param), imgs_dev(nullptr){};
    camParams param; // 相机参数
    uchar *imgs_dev; // 设备上的图像数据
};

// 数据加载器类
class DataLoader
{
  public:
    DataLoader() = default;
    // 数据加载器构造函数
    DataLoader(int _n_img,                                 // 相机数量
               int _h,                                     // 图像高度
               int _w,                                     // 图像宽度
               const std::string &_data_infos_path,        // 数据信息文件路径
               const std::vector<std::string> &_cams_name, // 相机名称列表
               bool _sep = true);                          // 是否分离加载

    // 获取相机内参矩阵
    const std::vector<Eigen::Matrix3f> &get_cams_intrin() const
    {
        return cams_intrin;
    }
    // 获取相机到自车旋转
    const std::vector<Eigen::Quaternion<float>> &get_cams2ego_rot() const
    {
        return cams2ego_rot;
    }
    // 获取相机到自车平移
    const std::vector<Eigen::Translation3f> &get_cams2ego_trans() const
    {
        return cams2ego_trans;
    }
    // 获取LiDAR到自车旋转
    const Eigen::Quaternion<float> &get_lidar2ego_rot() const
    {
        return lidar2ego_rot;
    }
    // 获取LiDAR到自车平移
    const Eigen::Translation3f &get_lidar2ego_trans() const
    {
        return lidar2ego_trans;
    }
    // 获取样本数量
    int size()
    {
        return sample_num;
    }
    // 获取指定索引的数据
    const camsData &data(int idx, bool time_order = true);
    ~DataLoader();

  private:
    std::vector<int> time_sequence; // 时间序列
    std::string data_infos_path;    // 数据信息文件路径
    int sample_num;                 // 样本数量

    std::vector<std::string> cams_name; // 相机名称列表
    int n_img;                          // 相机数量
    int img_h;                          // 图像高度
    int img_w;                          // 图像宽度

    std::vector<camParams> cams_param; // 相机参数列表
    camsData cams_data;                // 相机数据

    std::vector<Eigen::Matrix3f> cams_intrin;           // 相机内参矩阵
    std::vector<Eigen::Quaternion<float>> cams2ego_rot; // 相机到自车旋转
    std::vector<Eigen::Translation3f> cams2ego_trans;   // 相机到自车平移
    Eigen::Quaternion<float> lidar2ego_rot;             // LiDAR到自车旋转
    Eigen::Translation3f lidar2ego_trans;               // LiDAR到自车平移

#ifdef __HAVE_NVJPEG__
    nvjpegDecoder nvdecoder; // NVIDIA JPEG解码器
#endif
    uchar *imgs_dev;                          // 设备上的图像数据
    std::vector<std::vector<char>> imgs_data; // 主机上的图像数据
    bool separate;                            // 是否分离加载
};

// 从YAML节点读取平移向量
Eigen::Translation3f fromYamlTrans(YAML::Node x);
// 从YAML节点读取四元数
Eigen::Quaternion<float> fromYamlQuater(YAML::Node x);
// 从YAML节点读取3x3矩阵
Eigen::Matrix3f fromYamlMatrix3f(YAML::Node x);

// 读取图像文件
int read_image(std::string &image_names, std::vector<char> &raw_data);
// 读取样本数据
int read_sample(std::vector<std::string> &imgs_file,
                std::vector<std::vector<char>> &imgs_data);

#endif