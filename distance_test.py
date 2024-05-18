#import pyrealsense2 as rs
import numpy as np
import cv2
import random
#import torch
import time

import PIL
from numpy import asarray

import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image ,CompressedImage
# from sensor_msgs.msg import Image as msg_Image
import message_filters
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


import subprocess
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class FeatureDetectionROS():

    def __init__(self):
        rospy.init_node('rs_detection', anonymous=True)
        self.rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1, buff_size=9000000)
        # self.rgb_sub = message_filters.Subscriber("/camera/color/image_raw/compressed", CompressedImage, queue_size=1, buff_size=10000)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1, buff_size=9000000)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub , self.depth_sub], 1, 1)
        self.ts.registerCallback(self.frame_callback)
        self.bridge = CvBridge()

        self.featured_im_pub = rospy.Publisher('featured_image', Image, queue_size=10)

        # Intrensics
        self.K = np.array([
                            [605.2021484375, 0.0, 323.37109375],
                            [0.0, 605.3436279296875, 246.51541137695312],
                            [0.0, 0.0, 1.0]
                        ])
        self.fastFeatures = cv2.FastFeatureDetector_create()

        print('finish init')
        rospy.spin()

    # # Callback fucntion to convert ros_msg into image format
    # def frame_callback(self, rgb_data, depth_data):
    #     img = self.bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
    #     # cv2.imwrite('rgb.png', img)
    #     depth = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
    #     # cv2.imwrite('depth.png', depth)


    def frame_callback(self, rgb_data, depth_data):
        try:
            img = self.bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")

            # 获取图像中心点的坐标
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

            # 使用FAST特征检测器
            keypoints = self.fastFeatures.detect(img, None)

            # 如果没有检测到特征点，直接返回
            if not keypoints:
                rospy.loginfo("No keypoints detected.")
                return

            # 找到距离图像中心最近的特征点
            closest_keypoint = min(keypoints, key=lambda kp: (kp.pt[0] - center_x) ** 2 + (kp.pt[1] - center_y) ** 2)
            kp_x, kp_y = int(closest_keypoint.pt[0]), int(closest_keypoint.pt[1])

            # 获取特征点的深度值
            kp_depth = depth[kp_y, kp_x]

            # 在图像上标记特征点和深度信息
            text = f"Depth: {kp_depth}mm"
            cv2.circle(img, (kp_x, kp_y), 5, (0, 255, 0), -1)  # Draw a circle around the keypoint
            cv2.putText(img, text, (kp_x + 10, kp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 将标注了深度信息的图像转换为ROS消息
            featured_image_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            
            # 设置消息头
            featured_image_msg.header = rgb_data.header
            
            # 发布消息
            self.featured_im_pub.publish(featured_image_msg)
            
            print(f"Published featured image with keypoint depth: {kp_depth} at ({kp_x}, {kp_y})")
        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}")




    def draw_keypoints_and_lines(self, img, keypoints, lines=None):
        """绘制关键点和线条到图像上"""
        for point in keypoints:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        if lines is not None:
            for pt1, pt2 in lines:
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 255), 2)
        img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.featured_im_pub.publish(img_msg)
        # if (len(keypoints)<3):
        #     input("Press Enter to continue...")
        return img


# Start the object detection node!
if __name__ == "__main__":

    detector = FeatureDetectionROS()