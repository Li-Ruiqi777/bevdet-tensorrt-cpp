import os
import os.path as osp
import numpy as np
from open3d_vis import Visualizer
import yaml
import argparse

# 颜色调色板定义 (RGB格式)
# 参考: https://www.rapidtables.com/web/color/RGB_Color.html
PALETTE = [
    [30, 144, 255],   # 道奇蓝
    [0, 255, 255],    # 青色
    [255, 215, 0],    # 金黄色
    [160, 32, 240],   # 紫色
    [3, 168, 158],    # 锰蓝
    [255, 0, 0],      # 红色
    [255, 97, 0],     # 橙色
    [0, 201, 87],     # 翠绿色
    [255, 153, 153],  # 粉色
    [255, 255, 0],    # 黄色
    [0, 0, 0],        # 黑色
]

def show_result_meshlab(vis, 
                        data,
                        result,
                        out_dir=None,
                        gt_bboxes=None, 
                        score_thr=0.0,
                        snapshot=False):
    """
    使用MeshLab显示3D检测结果
    
    参数:
        vis: Visualizer对象
        data: 点云数据
        result: 检测结果数组，包含预测框、分数和标签
        out_dir: 输出目录(可选)
        gt_bboxes: 真实标注框(可选)
        score_thr: 分数阈值，过滤低分检测框
        snapshot: 是否截图(未实现)
    """
    points = data  # 点云数据
    pred_bboxes = result[:, :7]  # 预测框 (x,y,z,l,w,h,theta)
    pred_scores = result[:, 7]   # 预测分数
    pred_labels = result[:, 8]   # 预测标签

    # 根据分数阈值过滤低分检测框
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]
        pred_labels = pred_labels[inds]
    print('可视化对象数量: {} (分数阈值: {})'.format(pred_bboxes.shape[0], score_thr))

    # 清除现有几何体
    vis.o3d_visualizer.clear_geometries()
    # 添加点云
    p = np.random.rand(1, 5)
    # vis.add_points(points)
    vis.add_points(p)
    
    # 添加真实标注框(绿色)
    if gt_bboxes is not None:
        vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 1, 0), 
                       points_in_box_color=(0.5, 0.5, 0.5))
    
    # 添加预测框
    if pred_bboxes is not None:
        if pred_labels is None:
            vis.add_bboxes(bbox3d=pred_bboxes)
        else:
            # 按标签分类预测框
            labelDict = {}
            for j in range(len(pred_labels)):
                i = int(pred_labels[j])
                if labelDict.get(i) is None:
                    labelDict[i] = []
                labelDict[i].append(pred_bboxes[j])
            
            # 为不同标签的预测框设置不同颜色
            for i in labelDict:
                palette = [c / 255.0 for c in PALETTE[i]]  # 归一化到[0,1]
                vis.add_bboxes(
                    bbox3d=np.array(labelDict[i]),
                    bbox_color=palette, 
                    points_in_box_color=palette)

    # 设置相机视角
    ctr = vis.o3d_visualizer.get_view_control()
    ctr.set_lookat([0,0,0])       # 看向原点
    ctr.set_front([-1,-1,1])     # 设置垂直指向屏幕外的向量
    ctr.set_up([0,0,1])          # 设置指向屏幕上方的向量
    ctr.set_zoom(0.2)           # 设置缩放级别

    # 更新渲染
    vis.o3d_visualizer.poll_events()
    vis.o3d_visualizer.update_renderer()

def dataloader(cloud_path, boxes_path, load_dim):
    """
    加载点云和检测框数据
    
    参数:
        cloud_path: 点云文件路径
        boxes_path: 检测框文件路径
        load_dim: 点云维度
        
    返回:
        tuple: (检测结果, 点云数据)
    """
    # 加载点云数据
    data = np.fromfile(cloud_path, dtype=np.float32, count=-1).reshape([-1, load_dim])
    # 加载检测框数据
    result = np.loadtxt(boxes_path).reshape(-1, 9)
    return result, data

# 命令行参数解析
parser = argparse.ArgumentParser(description='3D检测结果可视化工具')
parser.add_argument("--score_thr", type=float, default=0.2,
                    help='可视化分数阈值，默认0.2')
parser.add_argument("--config", type=str, default="../configure.yaml",
                    help='配置文件路径，默认../configure.yaml')

args = parser.parse_args()

def main():
    """主函数"""
    # 加载配置文件
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 加载数据
    result, data = dataloader(config['InputFile'], config['OutputLidarBox'], config['LoadDim'])
    print(f'加载点云数据维度: {data.shape}')

    # 初始化可视化器
    vis = Visualizer(None)
    gt_bboxes = None  # 暂无真实标注
    
    # 显示结果
    show_result_meshlab(
        vis, 
        data,
        result,
        out_dir=None,
        gt_bboxes=gt_bboxes, 
        score_thr=args.score_thr,
        snapshot=False)
    
    # 显示窗口
    vis.show()

if __name__ == "__main__":
    main()