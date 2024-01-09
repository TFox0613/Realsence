import pyrealsense2 as rs
import cv2
import numpy as np
import datetime as dt
import mediapipe as mp


class rs_para:
    fps = 30
    res_x = 640
    res_y = 480
    mode = "control" # rgb, background, fusion, sg3i, control
    background_removed_color = 153 # Grey
    depth_scale = 1.0
    clipping_distance_in_meters = 2.0
    clipping_distance = 0.0
    Color_cx = 307.931
    Color_cy = 239.032
    Color_fx = 381.181
    Color_fy = 380.786
    Depth_cx = 318.487
    Depth_cy = 238.795
    Depth_fx = 389.837
    Depth_fy = 389.837
    
class plot_word_para:
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 100)
    fontScale = .5
    color = (0,50,255)
    thickness = 1
# ====== Mediapipe ======
def mp_init():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    return mpHands,hands,mpDraw

# ====== Realsense ======

def rs_camera_init(device_serial_num,rs_para):
    device = device_serial_num # In this example we are only using one camera
    pipeline = rs.pipeline()
    config = rs.config()
    # ====== Enable Streams ======
    config.enable_device(device)

    stream_res_x = rs_para.res_x
    stream_res_y = rs_para.res_y
    stream_fps = rs_para.fps
    print(stream_res_x,stream_res_y,stream_fps)

    config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
    config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
    
    profile = pipeline.start(config)
    # ====== Get depth Scale ======
    depth_sensor = profile.get_device().first_depth_sensor()
    rs_para.depth_scale = depth_sensor.get_depth_scale()
    rs_para.clipping_distance = rs_para.clipping_distance_in_meters / rs_para.depth_scale
    print(f"Depth Scale for Camera SN {device} is: {rs_para.depth_scale}")
    # ====== Show Camera Info ====== 
    depth_profile = profile.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    depth_intr = depth_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    print("Depth Intrisic : ",depth_intr)
    color_profile = profile.get_stream(rs.stream.color) # Fetch stream profile for depth stream
    color_intr = color_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    print("Color Intrisic : ",color_intr)
    rs_para.Depth_cx = depth_intr.ppx
    rs_para.Depth_cy = depth_intr.ppy
    rs_para.Depth_fx = depth_intr.fx
    rs_para.Depth_fy = depth_intr.fy
    rs_para.Color_cx = color_intr.ppx
    rs_para.Color_cy = color_intr.ppy
    rs_para.Color_fx = color_intr.fx
    rs_para.Color_fy = color_intr.fy

    print(rs_para)


    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline,align

def image_preprocessing(camera_pipline,camera_align,rs_para,plot_word_para):
    camera_frames = camera_pipline.wait_for_frames()
    camera_aligned_frames = camera_align.process(camera_frames)
    aligned_depth_frame = camera_aligned_frames.get_depth_frame()
    color_frame = camera_aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    finger_pos = [0,0,0]
    if rs_para.mode == "rgb":
        color_image = cv2.flip(color_image,1)
        return color_image , aligned_depth_frame
    
    if rs_para.mode == "background":
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_flipped = cv2.flip(depth_image,1)
        
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3
        background_removed = np.where((depth_image_3d > rs_para.clipping_distance) | (depth_image_3d <= 0), rs_para.background_removed_color, color_image)
        background_removed_images = background_removed
        background_removed_images = cv2.flip(background_removed,1)
        return background_removed_images , aligned_depth_frame
    
    if rs_para.mode == "control":
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # depth_image_flipped = cv2.flip(depth_image,1)
        # color_image = cv2.flip(color_image,1)
        return color_image,depth_image
    
    if rs_para.mode == "fusion":
        # Process hands
        [mpHands,hands,mpDraw] = mp_init()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_flipped = cv2.flip(depth_image,1)
        color_image = cv2.flip(color_image,1)
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(color_images_rgb)
        if results.multi_hand_landmarks:
            number_of_hands = len(results.multi_hand_landmarks)
            i=0
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(color_image, handLms, mpHands.HAND_CONNECTIONS)
                org2 = (20, plot_word_para.org[1]+(20*(i+1)))
                hand_side_classification_list = results.multi_handedness[i]
                hand_side = hand_side_classification_list.classification[0].label
                #==============middle finger knuckle (x,y)==============
                # middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]
                # x = int(middle_finger_knuckle.x*len(depth_image_flipped[0]))
                # y = int(middle_finger_knuckle.y*len(depth_image_flipped))
                
                #======================wrist (x,y)======================
                wrist = results.multi_hand_landmarks[i].landmark[8]
                x = int(wrist.x*len(depth_image_flipped[0]))
                y = int(wrist.y*len(depth_image_flipped))
                if x >= len(depth_image_flipped[0]):
                    x = len(depth_image_flipped[0]) - 1
                if y >= len(depth_image_flipped):
                    y = len(depth_image_flipped) - 1
                mfk_distance = depth_image_flipped[y,x] * rs_para.depth_scale # meters
                mfk_distance_feet = mfk_distance * 3.281 # feet
                #=================2D to 3D(RGB Camera)===================
                Xc = ((x-rs_para.Color_cx)/rs_para.Color_fx) * depth_image_flipped[y,x] * rs_para.depth_scale
                Yc = ((y-rs_para.Color_cy)/rs_para.Color_fy) * depth_image_flipped[y,x] * rs_para.depth_scale
                Zc = depth_image_flipped[y,x] * rs_para.depth_scale
                # images = cv2.putText(images, f"{hand_side} Hand Distance: {mfk_distance_feet:0.3} feet ({mfk_distance:0.3} m) away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
                color_image = cv2.putText(color_image, f"{hand_side} Hand Distance:({Xc:0.3} m), ({Yc:0.3} m), ({Zc:0.3} m) away", org2,\
                                           plot_word_para.font, plot_word_para.fontScale,\
                                              plot_word_para.color, plot_word_para.thickness, cv2.LINE_AA)
                finger_pos = [Xc,Yc,Zc]
                i+=1
            color_image = cv2.putText(color_image, f"Hands: {number_of_hands}"\
                                            , plot_word_para.org, plot_word_para.font, plot_word_para.fontScale,\
                                                    plot_word_para.color, plot_word_para.thickness, cv2.LINE_AA)
            lines = [str(Xc),'\t',str(Yc),'\t',str(Zc),'\r']
            hand_flag = True
        else:
            color_image = cv2.putText(color_image,"No Hands"\
                                        , plot_word_para.org, plot_word_para.font, plot_word_para.fontScale,\
                                            plot_word_para.color, plot_word_para.thickness, cv2.LINE_AA)
            hand_flag = False
             
        return color_image,finger_pos,hand_flag
    if rs_para.mode == "sg3i":
        # gugugugu
        return color_image,finger_pos
             
             
             

# [camera1_pipline,camera1_align] = rs_camera_init(connected_devices[0],rs_para)

if __name__ == '__main__':
    rs_para1 = rs_para()
    plot_word_para1 = plot_word_para()
    realsense_ctx = rs.context()
    connected_devices = [] # List of serial numbers for present cameras
    for i in range(len(realsense_ctx.devices)):
        detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        print(f"{detected_camera}")
        connected_devices.append(detected_camera)
    
    [camera1_pipline,camera1_align] = rs_camera_init(connected_devices[0],rs_para1)

    stop_flag = 0
    while not stop_flag:
        # Get and align frames
        # camera1_frames = camera1_pipline.wait_for_frames()
        # camera1_aligned_frames = camera1_align.process(camera1_frames)
        # aligned_depth_frame = camera1_aligned_frames.get_depth_frame()
        # color_frame = camera1_aligned_frames.get_color_frame()

        # color_image = np.asanyarray(color_frame.get_data())

        [image,finger_pos] = image_preprocessing(camera1_pipline,camera1_align,rs_para1,plot_word_para)
        
        name_of_window = "Camera 1"
        # Display images 
        cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name_of_window, image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print("End of the streaming ...")
            stop_flag = 1
