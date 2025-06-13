#ifndef __BEVDET_H__
#define __BEVDET_H__

#include <string>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common.h"
#include "postprocess.h"
#include "preprocess.h"
#include "data.h"

#include "NvInfer.h"

// TensorRT日志记录器
class Logger : public nvinfer1::ILogger {
 public:
  // 构造函数，可设置日志级别
  explicit Logger(Severity severity = Severity::kWARNING) : reportable_severity(severity){}

  // 日志记录方法
  void log(Severity severity, const char *msg) noexcept override {
    // 过滤低于设定级别的日志
    if (severity > reportable_severity) return;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportable_severity;  // 可报告的最低日志级别
};

// 存储相邻帧BEV特征数据的结构体
struct adjFrame{
    // 默认构造函数
    adjFrame(){}
    
    // 带参数的构造函数
    adjFrame(int _n,            // 存储的相邻帧数量
             int _map_size,      // BEV特征图大小(宽*高)
             int _bev_channel) : // BEV特征通道数
             n(_n), 
             map_size(_map_size), 
             bev_channel(_bev_channel),
             scenes_token(_n),   // 各帧的场景token
             ego2global_rot(_n), // 各帧的自车到全局旋转
             ego2global_trans(_n) { // 各帧的自车到全局平移
        CHECK_CUDA(cudaMalloc((void**)&adj_buffer, _n * _map_size * _bev_channel * sizeof(float)));
    }
    // 获取最后保存帧的场景token
    const std::string& lastScenesToken() const{
        return scenes_token[last];
    }

    // 重置帧缓冲区状态
    void reset(){
        last = -1;
        buffer_num = 0;
    }

    // 保存当前帧数据到缓冲区
    void saveFrameBuffer(const float* curr_buffer,        // 当前帧BEV特征数据
                        const std::string &curr_token,    // 当前场景token
                        const Eigen::Quaternion<float> &_ego2global_rot,  // 当前旋转
                        const Eigen::Translation3f &_ego2global_trans) {  // 当前平移
        last = (last + 1) % n;
        CHECK_CUDA(cudaMemcpy(adj_buffer + last * map_size * bev_channel, curr_buffer,
                        map_size * bev_channel * sizeof(float), cudaMemcpyDeviceToDevice));
        scenes_token[last] = curr_token;
        ego2global_rot[last] = _ego2global_rot;
        ego2global_trans[last] = _ego2global_trans;
        buffer_num = std::min(buffer_num + 1, n);
    }
    
    // 获取相对索引处帧的BEV特征数据
    const float* getFrameBuffer(int idx){
        idx = (-idx + last + n) % n;
        return adj_buffer + idx * map_size * bev_channel;
    }
    
    // 获取相对索引处帧的自车到全局变换
    void getEgo2Global(int idx, 
                      Eigen::Quaternion<float> &adj_ego2global_rot,  // 输出旋转
                      Eigen::Translation3f &adj_ego2global_trans) {  // 输出平移
        idx = (-idx + last + n) % n;
        adj_ego2global_rot = ego2global_rot[idx];
        adj_ego2global_trans = ego2global_trans[idx];
    }

    // 析构函数 - 释放CUDA内存
    ~adjFrame(){
        CHECK_CUDA(cudaFree(adj_buffer));
    }

    int n;              // 存储的相邻帧数量
    int map_size;       // BEV特征图大小(宽*高)
    int bev_channel;    // BEV特征通道数

    int last;           // 最后保存帧的索引
    int buffer_num;     // 当前缓冲区中的帧数

    std::vector<std::string> scenes_token;          // 各帧的场景token
    std::vector<Eigen::Quaternion<float>> ego2global_rot;    // 各帧的自车到全局旋转
    std::vector<Eigen::Translation3f> ego2global_trans;      // 各帧的自车到全局平移

    float* adj_buffer;   // 存储BEV特征的设备缓冲区
};

// BEVDet类 - 实现基于BEV(鸟瞰图)的3D目标检测
class BEVDet{
public:
    BEVDet(){}  // 默认构造函数
    // 带参数的构造函数
    BEVDet(const std::string &config_file,       // 配置文件路径
           int n_img,                            // 图像数量
           std::vector<Eigen::Matrix3f> _cams_intrin,  // 相机内参矩阵
           std::vector<Eigen::Quaternion<float>> _cams2ego_rot,  // 相机到自车旋转
           std::vector<Eigen::Translation3f> _cams2ego_trans,     // 相机到自车平移
           const std::string &imgstage_file,     // 图像阶段引擎文件
           const std::string &bevstage_file);    // BEV阶段引擎文件
  
    // 执行推理
    int DoInfer(const camsData &cam_data,        // 输入相机数据
                std::vector<Box> &out_detections, // 输出检测框
                float &cost_time,                // 耗时(毫秒)
                int idx=-1);                     // 帧索引(可选)

    ~BEVDet();  // 析构函数

protected:
    // 初始化参数
    void InitParams(const std::string &config_file);
    // 初始化视图变换器
    void InitViewTransformer();
    // 初始化推理引擎
    int InitEngine(const std::string &imgstage_file, const std::string &bevstage_file);
    // 反序列化TensorRT引擎
    int DeserializeTRTEngine(const std::string &engine_file, nvinfer1::ICudaEngine **engine_ptr);
    // 分配设备内存
    void MallocDeviceMemory();
    // 初始化深度估计
    void InitDepth(const std::vector<Eigen::Quaternion<float>> &curr_cams2ego_rot,
                   const std::vector<Eigen::Translation3f> &curr_cams2ego_trans,
                   const std::vector<Eigen::Matrix3f> &cams_intrin);

    // 获取相邻帧特征
    void GetAdjFrameFeature(const std::string &curr_scene_token,  // 当前场景token
                     const Eigen::Quaternion<float> &ego2global_rot,  // 自车到全局旋转
                     const Eigen::Translation3f &ego2global_trans,    // 自车到全局平移
                     float* bev_buffer);  // BEV特征缓冲区

    // 对齐BEV特征
    void AlignBEVFeature(const Eigen::Quaternion<float> &curr_ego2global_rot,  // 当前帧旋转
                         const Eigen::Quaternion<float> &adj_ego2global_rot,   // 相邻帧旋转
                         const Eigen::Translation3f &curr_ego2global_trans,  // 当前帧平移
                         const Eigen::Translation3f &adj_ego2global_trans,   // 相邻帧平移
                         const float* input_bev,    // 输入BEV特征
                         float* output_bev,         // 输出BEV特征
                         cudaStream_t stream);      // CUDA流

private:
    // 图像相关参数
    int N_img;              // 图像数量
    int src_img_h;          // 源图像高度
    int src_img_w;          // 源图像宽度
    int input_img_h;        // 输入图像高度
    int input_img_w;        // 输入图像宽度
    int crop_h;             // 裁剪高度
    int crop_w;             // 裁剪宽度
    float resize_radio;     // 缩放比例
    int down_sample;        // 下采样率
    int feat_h;             // 特征图高度
    int feat_w;             // 特征图宽度
    int bev_h;              // BEV高度
    int bev_w;              // BEV宽度
    int bevpool_channel;    // BEV池化通道数

    // 深度估计参数
    float depth_start;      // 深度起始值
    float depth_end;        // 深度结束值
    float depth_step;       // 深度步长
    int depth_num;          // 深度bin数量

    // 3D网格参数
    float x_start;          // X轴起始值
    float x_end;            // X轴结束值
    float x_step;           // X轴步长
    int xgrid_num;          // X轴网格数
    
    float y_start;          // Y轴起始值
    float y_end;            // Y轴结束值
    float y_step;           // Y轴步长
    int ygrid_num;          // Y轴网格数
    
    float z_start;          // Z轴起始值
    float z_end;            // Z轴结束值
    float z_step;           // Z轴步长
    int zgrid_num;          // Z轴网格数

    // 图像归一化参数
    triplet mean;           // 均值
    triplet std;            // 标准差

    // 功能开关
    bool use_depth;         // 是否使用深度估计
    bool use_adj;           // 是否使用相邻帧
    int adj_num;            // 相邻帧数量

    Sampler pre_sample;     // 预处理采样器

    // 检测相关参数
    int class_num;                  // 类别数量
    float score_thresh;             // 分数阈值
    float nms_overlap_thresh;        // NMS重叠阈值
    int nms_pre_maxnum;              // NMS前最大保留数
    int nms_post_maxnum;             // NMS后最大保留数
    std::vector<float> nms_rescale_factor;  // NMS重缩放因子
    std::vector<int> class_num_pre_task;    // 各任务类别数
    std::map<std::string, int> out_num_task_head;  // 各任务头输出数量

    // 相机参数
    std::vector<Eigen::Matrix3f> cams_intrin;      // 相机内参
    std::vector<Eigen::Quaternion<float>> cams2ego_rot;   // 相机到自车旋转
    std::vector<Eigen::Translation3f> cams2ego_trans;     // 相机到自车平移

    // 后处理变换
    Eigen::Matrix3f post_rot;       // 后处理旋转
    Eigen::Translation3f post_trans; // 后处理平移

    // 设备内存指针
    uchar* src_imgs_dev;            // 源图像设备内存
    void** imgstage_buffer;         // 图像阶段缓冲区
    void** bevstage_buffer;         // BEV阶段缓冲区

    // 缓冲区映射
    std::map<std::string, int> imgbuffer_map;  // 图像缓冲区映射
    std::map<std::string, int> bevbuffer_map;  // BEV缓冲区映射

    // 特征相关
    int valid_feat_num;             // 有效特征数
    int unique_bev_num;             // 唯一BEV特征数

    // CUDA设备内存
    int* ranks_bev_dev;             // BEV排序索引
    int* ranks_depth_dev;           // 深度排序索引
    int* ranks_feat_dev;            // 特征排序索引
    int* interval_starts_dev;       // 区间起始索引
    int* interval_lengths_dev;      // 区间长度

    // TensorRT相关
    Logger g_logger;                            // 日志记录器
    nvinfer1::ICudaEngine* imgstage_engine;     // 图像阶段引擎
    nvinfer1::ICudaEngine* bevstage_engine;     // BEV阶段引擎
    nvinfer1::IExecutionContext* imgstage_context;  // 图像阶段上下文
    nvinfer1::IExecutionContext* bevstage_context;  // BEV阶段上下文

    // 智能指针
    std::unique_ptr<PostprocessGPU> postprocess_ptr;  // 后处理指针
    std::unique_ptr<adjFrame> adj_frame_ptr;          // 相邻帧指针

};

// 数据类型到字节大小转换
__inline__ size_t dataTypeToSize(nvinfer1::DataType dataType);


#endif