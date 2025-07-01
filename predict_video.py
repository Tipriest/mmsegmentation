import cv2
import numpy as np
import os
import tempfile
from tqdm import tqdm
from mmseg.apis import init_model, inference_model
import matplotlib.pyplot as plt

def create_colored_mask(prediction, palette):
    """
    将预测结果转换为彩色掩码
    
    Args:
        prediction: 预测结果 (H, W)
        palette: 调色板 [(R, G, B), ...]
    
    Returns:
        colored_mask: 彩色掩码 (H, W, 3)
    """
    h, w = prediction.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(palette):
        mask = (prediction == class_id)
        colored_mask[mask] = color
    
    return colored_mask

def hex_to_bgr_palette(palette_rgb):
    """
    将十六进制RGB调色板转换为BGR格式（用于OpenCV）
    
    Args:
        palette_rgb: 十六进制RGB颜色列表
    
    Returns:
        palette: BGR格式的调色板列表
    """
    palette = []
    for hex_color in palette_rgb:
        r = (hex_color >> 16) & 0xFF
        g = (hex_color >> 8) & 0xFF  
        b = hex_color & 0xFF
        palette.append([b, g, r])  # BGR格式用于OpenCV
    return palette

def predict_video(config_path, checkpoint_path, video_path, output_path, alpha=0.5):
    """
    对视频进行语义分割预测并输出叠加预测结果的视频
    
    Args:
        config_path: 模型配置文件路径
        checkpoint_path: 模型权重文件路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        alpha: 叠加透明度，范围0-1，值越小原图越明显
    """
    # 初始化模型
    print("Loading model...")
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    
    # 获取调色板 (RGB格式的十六进制颜色)
    palette_rgb = [
        0xBBFFFF,  # concreteroad - 浅蓝色
        0x00FF00,  # road_curb - 绿色
        0xFFE4B5,  # redbrickroad - 浅黄色
        0x4169E1,  # zebracrossing - 蓝色
        0xFFFF00,  # stone_pier - 黄色
        0x9932CC,  # soil - 紫色
        0xFF1493,  # yellowbrick_road - 深粉色
    ]
    
    # 转换为BGR格式
    palette = hex_to_bgr_palette(palette_rgb)
    
    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 创建临时目录用于保存中间结果
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_count = 0
        
        # 处理每一帧
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存当前帧到临时文件
                temp_frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(temp_frame_path, frame)
                
                # 进行推理
                result = inference_model(model, temp_frame_path)
                
                # 获取预测结果
                if hasattr(result, 'pred_sem_seg'):
                    prediction = result.pred_sem_seg.data.cpu().numpy().squeeze()
                else:
                    prediction = result.prediction.cpu().numpy().squeeze()
                
                # 调整预测结果尺寸以匹配原始帧
                if prediction.shape != (height, width):
                    prediction = cv2.resize(prediction.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                
                # 创建彩色掩码
                colored_mask = create_colored_mask(prediction, palette)
                
                # 将结果与原图叠加
                blended = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
                
                # 写入输出视频
                out.write(blended)
                
                frame_count += 1
                pbar.update(1)
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed! Output saved to: {output_path}")

def predict_single_image(config_path, checkpoint_path, img_path, output_path=None, alpha=0.5):
    """
    对单张图像进行预测
    
    Args:
        config_path: 模型配置文件路径
        checkpoint_path: 模型权重文件路径
        img_path: 输入图像路径
        output_path: 输出图像路径（可选）
        alpha: 叠加透明度
    """
    # 初始化模型
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    
    # 推理
    result = inference_model(model, img_path)
    
    # 读取原图
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    
    # 获取预测结果
    if hasattr(result, 'pred_sem_seg'):
        prediction = result.pred_sem_seg.data.cpu().numpy().squeeze()
    else:
        prediction = result.prediction.cpu().numpy().squeeze()
    
    # 调整预测结果尺寸
    if prediction.shape != (height, width):
        prediction = cv2.resize(prediction.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    
    # 调色板 (从RGB十六进制转换为BGR)
    palette_rgb = [
        0xBBFFFF,  # concreteroad - 浅蓝色
        0x00FF00,  # road_curb - 绿色
        0xFFE4B5,  # redbrickroad - 浅黄色
        0x4169E1,  # zebracrossing - 蓝色
        0xFFFF00,  # stone_pier - 黄色
        0x9932CC,  # soil - 紫色
        0xFF1493,  # yellowbrick_road - 深粉色
    ]
    
    # 转换为BGR格式
    palette = hex_to_bgr_palette(palette_rgb)
    
    # 创建彩色掩码
    colored_mask = create_colored_mask(prediction, palette)
    
    # 叠加结果
    blended = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, blended)
        print(f"Result saved to: {output_path}")
    
    return result, blended

def extract_sample_frames(video_path, output_dir, num_frames=10):
    """
    从视频中提取样本帧用于测试
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        num_frames: 提取帧数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Extracted frame {frame_idx} -> {output_path}")
    
    cap.release()

if __name__ == "__main__":
    # 配置路径
    config_path = 'configs/segformer/segformer_mit-b0_8xb2-160k_tib_ground1-512x512.py'
    checkpoint_path = 'work_dirs/segformer_mit-b0_8xb2-160k_tib_ground1-512x512/iter_20000.pth'
    
    # 选择运行模式
    mode = "video"  # "video", "image", "extract_frames"
    
    if mode == "video":
        # 视频预测
        video_path = "/data/tib_ground1/origin_videos/test_videos/test_video1.mp4"  # 修改为您的输入视频路径
        output_base_path = "/data/tib_ground1/predict_result/"
        
        # 自动生成输出文件名：输入视频名_模型权重名_批次信息.mp4
        def generate_output_filename(video_path, checkpoint_path, output_base_path):
            # 提取输入视频名（无扩展名）
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # 从权重路径提取模型信息和iteration信息
            checkpoint_dir = os.path.basename(os.path.dirname(checkpoint_path))
            checkpoint_file = os.path.basename(checkpoint_path)
            
            # 简化模型名称：提取主要部分
            if 'segformer' in checkpoint_dir.lower():
                model_name = 'segformer_mit-b0'
            else:
                # 通用提取方法：取第一个和第二个部分
                parts = checkpoint_dir.split('_')
                if len(parts) >= 2:
                    model_name = f"{parts[0]}_{parts[1]}"
                else:
                    model_name = parts[0] if parts else 'model'
            
            # 提取iteration信息 (如 iter_20000)
            if 'iter_' in checkpoint_file:
                iter_info = checkpoint_file.split('.')[0]  # 获取 iter_20000
            else:
                iter_info = os.path.splitext(checkpoint_file)[0]
            
            # 生成输出文件名：inputVideo_modelName_iterInfo.mp4
            output_filename = f"{video_name}_{model_name}_{iter_info}.mp4"
            output_path = os.path.join(output_base_path, output_filename)
            
            return output_path
        
        # 检查输入视频是否存在
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            print("Please update the video_path variable with your input video file.")
        else:
            # 生成自动输出路径
            output_path = generate_output_filename(video_path, checkpoint_path, output_base_path)
            print(f"Output video will be saved to: {output_path}")
            predict_video(config_path, checkpoint_path, video_path, output_path, alpha=0.6)
    
    elif mode == "image":
        # 单图像预测
        img_path = 'data/tib_ground1/images/train/0052.jpg'
        output_base_path = 'work_dirs/'
        
        # 自动生成输出文件名：输入图像名_模型权重名_批次信息.png
        def generate_image_output_filename(img_path, checkpoint_path, output_base_path):
            # 提取输入图像名（无扩展名）
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 从权重路径提取模型信息和iteration信息
            checkpoint_dir = os.path.basename(os.path.dirname(checkpoint_path))
            checkpoint_file = os.path.basename(checkpoint_path)
            
            # 简化模型名称：提取主要部分
            if 'segformer' in checkpoint_dir.lower():
                model_name = 'segformer_mit-b0'
            else:
                # 通用提取方法：取第一个和第二个部分
                parts = checkpoint_dir.split('_')
                if len(parts) >= 2:
                    model_name = f"{parts[0]}_{parts[1]}"
                else:
                    model_name = parts[0] if parts else 'model'
            
            # 提取iteration信息 (如 iter_20000)
            if 'iter_' in checkpoint_file:
                iter_info = checkpoint_file.split('.')[0]  # 获取 iter_20000
            else:
                iter_info = os.path.splitext(checkpoint_file)[0]
            
            # 生成输出文件名：inputImage_modelName_iterInfo.png
            output_filename = f"{img_name}_{model_name}_{iter_info}.png"
            output_path = os.path.join(output_base_path, output_filename)
            
            return output_path
        
        if os.path.exists(img_path):
            output_path = generate_image_output_filename(img_path, checkpoint_path, output_base_path)
            print(f"Output image will be saved to: {output_path}")
            predict_single_image(config_path, checkpoint_path, img_path, output_path, alpha=0.6)
        else:
            print(f"Image file not found: {img_path}")
    
    elif mode == "extract_frames":
        # 从视频提取帧
        video_path = "input_video.mp4"
        output_dir = "work_dirs/sample_frames"
        
        if os.path.exists(video_path):
            extract_sample_frames(video_path, output_dir, num_frames=10)
        else:
            print(f"Video file not found: {video_path}")
