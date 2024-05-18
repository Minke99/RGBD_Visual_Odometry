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
        # 创建用于发布轨迹的publisher
        self.path_pub = rospy.Publisher('/vo_path', Path, queue_size=10)
        # 初始化用于存储轨迹的Path
        self.path = Path()
        self.path.header.frame_id = 'map'  # 通常为'map'或'odom'
        self.current_position = [0.0, 0.0, 0.0]  # X, Y, Z
        # Pub detection infor and modified image out
        # self.yolo_pub = rospy.Publisher('people', Float32MultiArray, queue_size=1)

        # Intrensics
        self.K = np.array([
                            [605.2021484375, 0.0, 323.37109375],
                            [0.0, 605.3436279296875, 246.51541137695312],
                            [0.0, 0.0, 1.0]
                        ])
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        plt.ion()
        # # Create a figure to display the images
        # self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 10))

        self.last_im = []
        self.last_dep = []

        print('finish init')
        rospy.spin()

    # Callback fucntion to convert ros_msg into image format
    def frame_callback(self, rgb_data, depth_data):
        img = self.bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
        # cv2.imwrite('rgb.png', img)
        depth = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
        # cv2.imwrite('depth.png', depth)
        deltaR = np.eye(3)

        R1 = np.eye(3)

        if not(np.shape(self.last_im) == (480,640,3)):
            self.last_im = img
            self.last_dep = depth
            print('init frame')
        else:
            h, w, *_ = img.shape
            kp1 = self.get_tiled_keypoints(self.last_im, 64, 48)
            tp1, tp2 = self.track_keypoints(self.last_im, img, kp1)
            Q1, Q2 = self.calc_3d(tp1, self.last_dep, tp2, depth)
            Q1, Q2 = self.filter_far(Q1, Q2, 200)
            offset = self.estimate_trans(Q1, Q2, R1, deltaR)
            self.last_im = img
            self.last_dep = depth


    def estimate_trans(self, Q1, Q2, R1, deltaR):

        def ransac_translation(Q1, Q2, sample_ratio=0.4, num_iterations=500, std_dev_factor=1):
            if Q1.size == 0 or Q2.size == 0:
                raise ValueError("Input arrays Q1 and Q2 must not be empty.")

            n_points = Q1.shape[0]
            if n_points == 0:
                raise ValueError("Input arrays Q1 and Q2 must contain points.")

            num_samples = int(n_points * sample_ratio)
            translations = []

            for _ in range(num_iterations):
                # 随机选择数据点的索引
                indices = np.random.choice(n_points, num_samples, replace=False)
                sample_Q1 = Q1[indices]
                sample_Q2 = Q2[indices]

                # 计算所有选定点对的平均位移
                sample_translations = sample_Q2 - sample_Q1
                mean_translation = np.mean(sample_translations, axis=0)
                std_dev = np.std(sample_translations, axis=0)

                # 筛选出与平均位移在一定标准差范围内的位移
                valid_translations = sample_translations[np.all(np.abs(sample_translations - mean_translation) < std_dev_factor * std_dev, axis=1)]
                if valid_translations.size > 0:
                    translations.extend(valid_translations.tolist())

            # 将收集到的所有有效位移转换为numpy数组，便于计算
            if translations:
                translations = np.array(translations)
                final_translation = np.mean(translations, axis=0)
            else:
                raise Exception("No valid translations were found. Try adjusting the parameters.")

            return final_translation
        
        # RQ2 = np.dot(deltaR, Q2)
        Q2_transposed = Q2.T
        RQ2_transposed = np.dot(deltaR, Q2_transposed)
        RQ2 = RQ2_transposed.T
        try:
            offset = ransac_translation(Q1, RQ2)
        except:
            offset=[0,0,0]
        
        Q1COM = np.mean(Q1, axis=0)
        Q2COM = np.mean(RQ2, axis=0)
        print(f"Q1COM [{Q1COM[0]:>7.1f},{Q1COM[1]:>7.1f}, {Q1COM[2]:>7.1f}] Q2COM [{Q2COM[0]:>7.1f},{Q2COM[1]:>7.1f},{Q2COM[2]:>7.1f}]")
        # offset = np.dot(R1, (Q1COM - RQ2COM))
        print(f"Trans Direction: [{offset[0]:>7.1f}, {offset[1]:>7.1f}, {offset[2]:>7.1f}], Num Points:{np.shape(Q1)[0]}")
        if np.shape(Q1)[0] >20:
            self.publish_trajectory(offset)
        return offset
    
    def publish_trajectory(self, offset):
        # 累积相对位移以更新当前位置
        self.current_position[0] += offset[0]
        self.current_position[1] += offset[1]
        self.current_position[2] += offset[2]

        # 创建PoseStamped消息
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose.position.x = self.current_position[0]
        pose_stamped.pose.position.y = self.current_position[1]
        pose_stamped.pose.position.z = self.current_position[2]
        pose_stamped.pose.orientation.w = 1.0  # 无旋转

        # 将新的位姿加入路径
        self.path.poses.append(pose_stamped)
        self.path.header.stamp = rospy.Time.now()

        # 发布路径
        self.path_pub.publish(self.path)


    def calc_3d(self, tp1, dep1, tp2, dep2):
        """
        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        f_x = self.K[0, 0]  # x轴焦距
        f_y = self.K[1, 1]  # y轴焦距
        c_x = self.K[0, 2]  # x轴光学中心
        c_y = self.K[1, 2]  # y轴光学中心

        Q1 = np.zeros((tp1.shape[0], 3))
        for i, (x, y) in enumerate(tp1):
            Z = dep1[int(y), int(x)]  # Assuming dep1 is a 2D array
            X = (x - c_x) * Z / f_x
            Y = (y - c_y) * Z / f_y
            Q1[i, :] = [X, Y, Z]

        Q2 = np.zeros((tp2.shape[0], 3))
        for i, (x, y) in enumerate(tp2):
            Z = dep2[int(y), int(x)]  # Assuming dep2 is also a 2D array
            X = (x - c_x) * Z / f_x
            Y = (y - c_y) * Z / f_y
            Q2[i, :] = [X, Y, Z]

        # self.plot_3d_points(Q1, Q2)
        return Q1, Q2

    def plot_3d_points(self, Q1, Q2):
        # 创建一个新的3D图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制Q1点集
        ax.scatter(Q1[:, 0], Q1[:, 1], Q1[:, 2], c='blue', marker='o', label='Q1 points')
        # 绘制Q2点集
        ax.scatter(Q2[:, 0], Q2[:, 1], Q2[:, 2], c='red', marker='^', label='Q2 points')

        Q1COM = np.mean(Q1, axis=0)
        Q2COM = np.mean(Q2, axis=0)

        ax.scatter(Q1COM[0], Q1COM[1], Q1COM[2], c='blue', marker='*', s=100, label='Q1 COM')  # 大蓝星
        ax.scatter(Q2COM[0], Q2COM[1], Q2COM[2], c='red', marker='*', s=100, label='Q2 COM')  # 大红星

        # 设置图例
        ax.legend()

        # 设置坐标轴标签
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')

        # 显示图形
        plt.show()
        input("Press Enter to continue...")

    def filter_far(self, Q1, Q2, dis = 300):
        # 确保 Q1 和 Q2 的形状相同
        assert Q1.shape == Q2.shape, "Q1 and Q2 must have the same shape"
        
        # 找出 Q1 和 Q2 中第三个元素（index 2）超过 30 的点的索引
        indices_to_keep = np.where((Q1[:, 2] <= dis) & (Q2[:, 2] <= dis))[0]
        
        # 过滤出保留的点
        Q1_filtered = Q1[indices_to_keep]
        Q2_filtered = Q2[indices_to_keep]
        
        return Q1_filtered, Q2_filtered


    def get_tiled_keypoints(self, img, tile_h, tile_w):
        """
        Splits the image into tiles and detects the 10 best keypoints in each tile
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fastFeatures.detect(impatch)

            # 创建 ORB 检测器，可以设置最大特征点数等参数
            # orb = cv2.ORB_create(nfeatures=2000,  # 增加特征点的数量
            #          scaleFactor=1.2,  # 调整金字塔的缩放系数，减小这个值可以增加图像金字塔的层数
            #          edgeThreshold=10,  # 减小边缘阈值，允许靠近边缘的特征点被检测
            #          fastThreshold=15,  # 降低 FAST 检测器的阈值，使其更敏感
            #          scoreType=cv2.ORB_HARRIS_SCORE)  # 改用 Harris score 来改善特征点的质量和稳定性
            # # 检测关键点和计算描述子
            # orb_keypoints, orb_descriptors = orb.detectAndCompute(impatch, None)
            # # 合并从 ORB 和 FAST 得到的关键点
            # keypoints = keypoints + orb_keypoints
            # print(len(orb_keypoints))


            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints
        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=10):
        """
        Tracks the keypoints between frames
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert keypoints into a vector of points and expand the dims so can select good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)


        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w, c = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        # self.plot_tracking(img1, img2, trackpoints1, trackpoints2)
        # Prepare lines to draw between matched keypoints
        lines = list(zip(trackpoints1, trackpoints2))
        # Draw keypoints and lines on the image
        self.draw_keypoints_and_lines(img2, trackpoints2, lines)

        return trackpoints1, trackpoints2

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