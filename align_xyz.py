import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# 1. 初始化管道
pipeline = rs.pipeline()
config = rs.config()
# 显式指定分辨率
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 加载模型
model = YOLO("best.pt")

# 2. 启动流
profile = pipeline.start(config)

# 初始化对齐和上色工具
align = rs.align(rs.stream.color)
colorizer = rs.colorizer()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        
        # 创建一个副本作为绘图画布，替代原来的 annotated_frame
        annotated_frame = color_image.copy()

        # 3. YOLO 推理
        results = model(color_image, conf=0.5) 
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 提取中心点 (用于测距) 和 左上右下点 (用于画框)
                    cx, cy, w, h = box.xywh[0].tolist()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 提取类别和置信度
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = model.names[cls]

                    # ----------------- 核心测距操作 -----------------
                    # 将中心点浮点坐标转换为整数索引
                    cx_int, cy_int = int(cx), int(cy)
                    
                    # 边界安全检查，防止越界崩溃 (根据你设置的 640x480 分辨率)
                    if 0 <= cx_int < 640 and 0 <= cy_int < 480:
                        # 获取该像素点的深度值，单位：米
                        distance = depth_frame.get_distance(cx_int, cy_int)
                    else:
                        distance = 0.0

                    # ----------------- 自定义渲染逻辑 -----------------
                    # A. 绘制边界框
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # B. 格式化标签字符串 (类别 置信度 深度)
                    label = f"{class_name} {conf:.2f} {distance:.2f}m"
                    
                    # C. 绘制文本背景框 (增强对比度，使文字清晰可见)
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1) - text_h - 10), (int(x1) + text_w, int(y1)), (0, 255, 0), -1)
                    
                    # D. 绘制文本
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 4. 深度图处理 (仅用于可视化展示)
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(depth_color_frame.get_data())

        # 拼接图像并显示
        images = np.hstack((annotated_frame, depth_image))
        cv2.imshow('RealSense YOLO Detection', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()