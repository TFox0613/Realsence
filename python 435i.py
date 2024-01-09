# test python 435i 
import pyrealsense2 as rs
import numpy as np
import cv2

# 對realsense 做連線
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30) #rs.format.z16 深度資料的bite
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)  


# 開始連接
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frames = frames.get_depth_frame()
        color_frames = frames.get_color_frame()
        # 將影像資訊轉換成array
        depth_image = np.asanyarray(depth_frames.get_data())
        color_image = np.asanyarray(color_frames.get_data())
        
        # 將深度資訊轉換為色彩的深度映射
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.03),cv2.COLORMAP_JET)
        #將深度合影向做並排

        images = np.vstack((color_image,depth_colormap))
        

        
        cv2.namedWindow('Realsense',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Realense',images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key==27:
            cv2.destroyAllWindows()
            break
except:
    print("Error")
    print("git v2 test")

finally:
    pipeline.stop()