# import pyrealsense2 as rs
# import cv2
# import numpy as np


# camera_depth_intrinsics          = rs.intrinsics()  # 相机深度内参
# camera_color_intrinsics          = rs.intrinsics()  # 相机彩色内参
# camera_depth_to_color_extrinsics = rs.extrinsics()  # 从深度相机到彩色相机的外参

# pipeline = rs.pipeline()

#     # 配置流数据
# config = rs.config()
# config.enable_stream(rs.stream.depth)
# config.enable_stream(rs.stream.color)

#     # 开始推流
# cfg = pipeline.start(config)

# # 创建深度对齐对象
# # rs.align 允许我们将深度帧对齐到其他帧
# # "align_to" 是指我们计划将深度帧对齐到的目标流类型
# # 这里将深度帧对齐到彩色帧
# align_to = rs.stream.color
# align = rs.align(align_to)


# end=True
# try:
#     while end:
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align.process(frames)
#         depth = frames.get_depth_frame()
#         color = frames.get_color_frame()
#         depth_image = np.asanyarray(depth.get_data())
#         color_image = np.asanyarray(color.get_data())


#         aligned_depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()
#         colorizer = rs.colorizer()
#         aligned_depth_frame = colorizer.colorize(aligned_depth_frame)   
#         npy_aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
#         npy_color_image = np.asanyarray(color_frame.get_data())

#         images = np.hstack((npy_aligned_depth_image, npy_color_image))
#         cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
#         cv2.imshow('Align Example', images)
#         key = cv2.waitKey(1)

#     if key ==ord('q'):
#         end=False
#         cv2.destroyAllWindows()
# finally:
#     # 5. 妥善释放资源
#     pipeline.stop()
#     cv2.destroyAllWindows()
import pyrealsense2 as rs
import cv2
import numpy as np

# 1. 初始化管道和配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 2. 启动流
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()

print("程序已启动，按 'q' 键退出...")

try:
    while True:
        # 等待每一帧数据
        frames = pipeline.wait_for_frames()
        # 对齐帧
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 转换为 numpy 数组
        # 深度图上色
        depth_color_frame = colorizer.colorize(aligned_depth_frame)
        depth_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 水平拼接显示
        images = np.hstack((color_image, depth_image))
        
        cv2.imshow('RealSense Align', images)

        # 4. 使用 ord() 转换字符，并配合 0xFF 掩码（保证 64 位系统兼容性）
        key = cv2.waitKey(1) # 这里用 1 毫秒，否则画面会卡死在第一帧
        if key & 0xFF == ord('q'):
            break

finally:
    # 5. 妥善释放资源
    pipeline.stop()
    cv2.destroyAllWindows()