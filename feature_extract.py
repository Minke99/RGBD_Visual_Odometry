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

import subprocess
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FeatureDetectionROS():

    def __init__(self):
        rospy.init_node('rs_detection', anonymous=True)
        self.rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1, buff_size=9000000)
        # self.rgb_sub = message_filters.Subscriber("/camera/color/image_raw/compressed", CompressedImage, queue_size=1, buff_size=10000)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1, buff_size=9000000)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub , self.depth_sub], 10, 150)
        self.ts.registerCallback(self.frame_callback)
        self.bridge = CvBridge()

        self.featured_im_pub = rospy.Publisher('featured_image', Image, queue_size=10)
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
            kp1 = self.get_tiled_keypoints(self.last_im, 10, 20)
            tp1, tp2 = self.track_keypoints(self.last_im, img, kp1)
            Q1, Q2 = self.calc_3d(tp1, self.last_dep, tp2, depth)
            offset = self.estimate_trans(Q1, Q2, R1, deltaR)
            self.last_im = img
            self.last_dep = depth


    def estimate_trans(self, Q1, Q2, R1, deltaR):

        def ransac_translation(Q1, Q2, num_iterations=1000, threshold=0.1):
            """
            使用RANSAC算法来估计并过滤在只有平移运动的情况下的3D点对。
            - num_iterations: int, RANSAC的迭代次数。
            - threshold: float, 确定内点的距离阈值。
            return:
            - best_translation: 最好的平移向量估计。
            - inliers: 内点的布尔索引数组。
            """
            best_inlier_count = 0
            best_translation = None
            best_inliers = None

            n_points = Q1.shape[0]

            for _ in range(num_iterations):
                # 随机选择一个点对来估计模型
                idx = np.random.randint(0, n_points)
                translation_estimate = Q2[idx] - Q1[idx]

                # 计算所有点对的平移向量
                estimated_Q2 = Q1 + translation_estimate
                distances = np.linalg.norm(estimated_Q2 - Q2, axis=1)
                
                # 确定内点
                inliers = distances < threshold
                
                # 更新最好的模型
                inlier_count = np.sum(inliers)
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_translation = translation_estimate
                    best_inliers = inliers

            return best_translation, best_inliers
        
        RQ2 = np.dot(deltaR, Q2)
        offset, inliers = ransac_translation(Q1, RQ2)
        
        # Q1COM = np.mean(Q1, axis=0)
        # Q2COM = np.mean(Q2, axis=0)
        # offset = np.dot(R1, (Q1COM - RQ2COM))
        print(f"Trans Direction: [{offset[0]:>7.1f}, {offset[1]:>7.1f}, {offset[2]:>7.1f}], Num Points:{np.shape(Q1)[0]}")

        return offset
        
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
        
    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames
        ----------
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
        self.draw_keypoints_and_lines(img2, trackpoints2)
        
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
        return img


# Start the object detection node!
if __name__ == "__main__":

    detector = FeatureDetectionROS()
