# 视频分割预测脚本使用说明

## 功能概述

`predict_video.py` 脚本支持三种运行模式：
1. **视频分割预测** - 对整个视频进行逐帧分割并输出结果视频
2. **单图像分割预测** - 对单张图像进行分割预测
3. **视频帧提取** - 从视频中提取样本帧用于测试

## 自动文件名生成

### 视频输出文件名格式
```
{输入视频名}_{模型名}_{批次信息}.mp4
```

**示例：**
- 输入视频：`original_video.mp4`
- 模型权重：`iter_20000.pth` (来自 segformer 模型)
- 输出文件：`original_video_segformer_mit-b0_iter_20000.mp4`

### 图像输出文件名格式
```
{输入图像名}_{模型名}_{批次信息}.png
```

**示例：**
- 输入图像：`0052.jpg`
- 模型权重：`iter_20000.pth`
- 输出文件：`0052_segformer_mit-b0_iter_20000.png`

## 使用方法

### 1. 配置文件路径
在脚本主函数中修改以下路径：
```python
config_path = 'configs/segformer/segformer_mit-b0_8xb2-160k_tib_ground1-512x512.py'
checkpoint_path = 'work_dirs/segformer_mit-b0_8xb2-160k_tib_ground1-512x512/iter_20000.pth'
```

### 2. 视频分割预测
```python
mode = "video"
video_path = "/data/tib_ground1/origin_videos/train_val_video/original_video.mp4"
output_base_path = "/data/tib_ground1/predict_result/"
```

运行命令：
```bash
python predict_video.py
```

输出示例：
```
Output video will be saved to: /data/tib_ground1/predict_result/original_video_segformer_mit-b0_iter_20000.mp4
```

### 3. 单图像分割预测
```python
mode = "image"
img_path = 'data/tib_ground1/images/train/0052.jpg'
output_base_path = 'work_dirs/'
```

输出示例：
```
Output image will be saved to: work_dirs/0052_segformer_mit-b0_iter_20000.png
```

### 4. 视频帧提取
```python
mode = "extract_frames"
video_path = "input_video.mp4"
output_dir = "work_dirs/sample_frames"
```

## 特性说明

### 自动目录创建
- 输出目录不存在时会自动创建
- 支持多级目录创建

### 进度显示
- 视频处理时显示实时进度条
- 显示处理速度（帧/秒）

### 颜色调色板支持
- 支持十六进制RGB格式调色板：`['#FF0000', '#00FF00', '#0000FF']`
- 自动转换为OpenCV需要的BGR格式

### 临时文件管理
- 自动清理处理过程中的临时文件
- 异常退出时也会清理临时文件

## 注意事项

1. **输入文件检查**：脚本会自动检查输入文件是否存在
2. **权重文件匹配**：确保配置文件与权重文件匹配
3. **GPU内存**：大视频处理时注意GPU内存使用
4. **输出格式**：
   - 视频输出：MP4格式，H.264编码
   - 图像输出：PNG格式，支持透明度

## 输出文件命名规则

### 模型名称简化规则
- SegFormer模型：`segformer_mit-b0`
- 其他模型：提取模型目录的前两个组件

### 批次信息提取规则
- 优先提取 `iter_` 前缀的信息（如 `iter_20000`）
- 其他情况使用完整的权重文件名（去掉扩展名）

## 示例输出文件名

| 输入文件 | 权重文件 | 输出文件 |
|---------|---------|---------|
| `video.mp4` | `iter_20000.pth` | `video_segformer_mit-b0_iter_20000.mp4` |
| `test.jpg` | `iter_40000.pth` | `test_segformer_mit-b0_iter_40000.png` |
| `sample.mp4` | `final_model.pth` | `sample_segformer_mit-b0_final_model.mp4` |

这样的命名方式确保了：
- 可以通过文件名快速识别输入来源
- 明确知道使用的是哪个模型和训练批次
- 避免不同配置之间的输出文件冲突
