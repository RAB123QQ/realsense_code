# from ultralytics import YOLO
# import pyrealsense2 as rs
# import cv2
# import numpy as np

# # 1. 初始化管道和配置
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# model = YOLO("best.pt")
# # 2. 启动流
# profile = pipeline.start(config)

# align_to = rs.stream.color
# align = rs.align(align_to)
# colorizer = rs.colorizer()

# print("程序已启动，按 'q' 键退出...")

# try:
#     while True:
#         # 等待每一帧数据
#         frames = pipeline.wait_for_frames()
#         # 对齐帧
#         aligned_frames = align.process(frames)
#         aligned_depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         if not aligned_depth_frame or not color_frame:
#             continue

#         # 转换为 numpy 数组
#         # 深度图上色（方便人类观察）
#         depth_color_frame = colorizer.colorize(aligned_depth_frame)
#         depth_image = np.asanyarray(depth_color_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data(),dtype=uint8)
#         results = model(color_image,show=True)
#         # 水平拼接显示
#         images = np.hstack((color_image, depth_image))
        
#         cv2.imshow('RealSense Align', images)
#         #cv2.imshow('RealSense Align YOLO', results)

#         # 4. 使用 ord() 转换字符，并配合 0xFF 掩码（保证 64 位系统兼容性）
#         key = cv2.waitKey(1) # 这里用 1 毫秒，否则画面会卡死在第一帧
#         if key & 0xFF == ord('q'):
#             break

# finally:
#     # 5. 妥善释放资源
#     pipeline.stop()
#     cv2.destroyAllWindows()
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# 1. 初始化管道
pipeline = rs.pipeline()
config = rs.config()
# 显式指定分辨率，确保数据流稳定
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
        
        # 3. YOLO 推理
        # stream=True 可以更高效地处理实时流
        results = model(color_image, conf=0.5) 
        
        # 将检测结果绘制到 color_image 上
        # results[0].plot() 会返回一个带有检测框的 BGR 图像 (uint8)
        annotated_frame = results[0].plot()

        # 4. 深度图处理
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(depth_color_frame.get_data())

        # 【原理确认】此时 annotated_frame 和 depth_image 都是 uint8，且维度相同
        # 只有这样 np.hstack 才是安全的
        images = np.hstack((annotated_frame, depth_image))
        
        cv2.imshow('RealSense YOLO Detection', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()