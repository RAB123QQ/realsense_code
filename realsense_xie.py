import torch
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import cv2
import time



pipeline = rs.pipeline()  # 定义流程 pipeline
config = rs.config()  # 定义配置 config
# config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与 color 流对齐
align = rs.align(align_to)

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的 depth 帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的 color 帧
    
    ############## 获取相机参数#######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }

    # 保存内参到本地
    with open('./intrinsics.json', 'w') as fp:
        json.dump(camera_parameters, fp)

    ######################################################
    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认 16 位）
    color_image = np.asanyarray(color_frame.get_data())  # RGB 图
    # 返回相机内参、深度参数、彩色图、深度图、对齐帧中的 depth 帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame




print("[INFO] 冬枣目标检测-程序启动")
print("[INFO] 开始 Yolo 模型加载")

print("[INFO] 完成 Yolo 模型加载")
try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
            # if not depth_image.any() or not color_image.any():
            # continue
            # Convert images to numpy arrays

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            t_start = time.time()  # 开始计时
            # YoloV5 目标检测

            t_end = time.time()  # 结束计时\
            # canvas = np.hstack((canvas, depth_colormap))
            # print(class_id_list)
            camera_xyz_list = []
            n = 0
            print('排序前：')
            print(camera_xyz_list)
            import numpy as np
            np.savetxt(str(n) + ".csv", camera_xyz_list, delimiter=",")
            camera_xyz_list.sort(reverse=False)
            # print('排序后：')
            # print(camera_xyz_list)
            # print('最近的 xyz 坐标位置:' + str(camera_xyz_list[0]) + 'm')
            zhixing(camera_xyz_list)
            # 添加 fps 显示
            fps = int(1.0 / (t_end - t_start))
            cv2.putText(canvas, text="FPS: {}".format(fps), org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                        lineType=cv2.LINE_AA, color=(0, 0, 0))
            cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('detection', canvas)
            key = cv2.waitKey(2000)
            cv2.imwrite(str(n) + '.jpg', canvas)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            if len(camera_xyz_list):
                cv2.destroyAllWindows()
                break
            # 如果能识别到果子则跳出循环
finally:
        # Stop streaming
        pipeline.stop()