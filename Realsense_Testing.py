import pyrealsense2 as rs
import numpy as np
import cv2
import time
 
if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    # 配置串流物件
    config = rs.config()
    # 宣告特定設備進行影像串流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            start = time.time()
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # 取得深度影像
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
 
            depth_image = np.asanyarray(depth_frame.get_data())
 
            color_image = np.asanyarray(color_frame.get_data())
 
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            time.sleep(0.01)
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(images,f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 1, cv2.LINE_AA)
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    except:
        print("Error")
    finally:
        # Stop streaming
        pipeline.stop()