# Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist


# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):
    img1_dig = int(np.sqrt(img1.shape[0] * 2 + img1.shape[1] * 2))
    img2_dig = int(np.sqrt(img2.shape[0] * 2 + img2.shape[1] * 2))

    space_left, space_right = int((img1.shape[1] - img1_dig) - 100), int((img1.shape[1] - img1_dig) - 100)
    space_top, space_bottom = int((img1.shape[0] - img1_dig) - 100), int((img1.shape[0] - img1_dig) - 100)
    print(space_left, space_right, space_top, space_bottom)

    img1_pad = np.pad(img1, ((space_left, space_right), (space_top, space_bottom), (0, 0)), mode='constant',
                      constant_values=0)

    cv2.imshow('img1_pad', img1_pad)
    cv2.waitKey(0)
    plt.show()

    img1_pad_dig = int(np.sqrt(img1_pad.shape[0] * 2 + img1_pad.shape[1] * 2))

    img1_warp = cv2.warpPerspective(img1_pad, M, (33 ** 2, 33 ** 2))
    img1_warp_copy = np.copy(img1_warp)

    sift = cv2.SIFT_create()

    k_w, d_w = sift.detectAndCompute(img1_warp, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    warped_matches = matching_keypoints(k_w, k2, d_w, d2)

    cv2.imshow('img1_warp', img1_warp)
    cv2.waitKey(0)
    plt.show()

    x_diff = []
    y_diff = []
    for match in warped_matches:
        x_diff.append(match[0] - match[2])
        y_diff.append(match[1] - match[3])
        x_mean = np.mean(x_diff)
        y_mean = np.mean(y_diff)

    img2_pad = np.pad(img2, ((int(y_mean), 0), (int(x_mean), 0), (0, 0)), mode='constant', constant_values=0)

    cv2.imshow('img2_pad', img2_pad)
    cv2.waitKey(0)
    plt.show()

    img1_warp[int(y_mean):img2_pad.shape[0], int(x_mean):img2_pad.shape[1]] = img2
    cv2.imshow('result', img1_warp)
    cv2.waitKey(0)
    plt.show()

    mask = np.zeros((img1_warp.shape[:2]), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (680, 450, 260, 310)
    cv2.grabCut(img1_warp, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_cut = img1_warp * mask2[:, :, np.newaxis]
    plt.imshow(img_cut), plt.colorbar(), plt.show()
    img1_warp = img1_warp - img_cut
    img1_warp = np.where(img1_warp == [0, 0, 0], img1_warp_copy, img1_warp)
    plt.imshow(img1_warp), plt.colorbar(), plt.show()

    return img1_warp

def matching_keypoints(kps1, kps2, desc1, desc2):

    pairwiseDistances = cdist(desc1, desc2, 'sqeuclidean')
    threshold = 5000

    points_in_img1 = np.where(pairwiseDistances < threshold)[0]
    points_in_img2 = np.where(pairwiseDistances < threshold)[1]

    coordinates_in_img1 = np.array([kps1[point].pt for point in points_in_img1])
    coordinates_in_img2 = np.array([kps2[point].pt for point in points_in_img2])

    return np.concatenate((coordinates_in_img1, coordinates_in_img2), axis=1)


# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Extract keypoints and descriptors
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    matches = matching_keypoints(k1, k2, d1, d2)
    print('matches', matches)
    print(matches[:, :2])

    m1 = np.copy(matches[:, :2])
    m2 = np.copy(matches[:, 2:4])

    # Compute homography matrix
    M, mask = cv2.findHomography(m1, m2, cv2.RANSAC, 5.0)
    print('M', M)
    return M


def stitch_background(img1, img2, savepath=''):
    """The output image should be saved in the savepath."""
    "Do NOT modify the code provided."

    # Use SIFT to find keypoints and return homography matrix
    M = get_sift_homography(img1, img2)

    # Stitch the images together and blend the images using grab cutter
    result_image = get_stitched_image(img1, img2, M)

    #Remove blur from the resulting image
    result_image = cv2.GaussianBlur(result_image, (5, 5), 0)

    cv2.imwrite('blend.png', result_image)
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    plt.show()

    return


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
