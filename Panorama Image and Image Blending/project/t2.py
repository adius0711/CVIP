# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):
    # Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimensions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    # Return the result
    return result_img


def matches_d(img1, img2):
    # Initialize SIFT
    sift = cv2.SIFT_create(1000)

    # Extract keypoints and descriptors
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    distances = np.sum(d1 ** 2, axis=1, keepdims=True) + np.sum(d2 ** 2, axis=1) - 2 * d1.dot(d2.T)
    distances = np.sqrt(distances)
    # print(distances)

    # Get smallest indices
    min_indices = np.argsort(distances, axis=1)

    # Init ndarray
    matches = []

    # Iter for nearest neighbors
    for i in range(min_indices.shape[0]):
        neighbors = min_indices[i][:2]
        # print(neighbors)
        curr_matches = []
        for j in range(len(neighbors)):
            match = []
            match.append(i)
            match.append(neighbors[j])
            match.append(distances[i][neighbors[j]] * 1.)
            curr_matches.append(match)
        matches.append(curr_matches)

        # Make sure that the matches are good
        verify_ratio = 0.8  # Source: stackoverflow
        verified_matches = []
        for m1, m2 in matches:
            # Add to array only if it's a good match
            if m1[2] < 0.8 * m2[2]:
                verified_matches.append(m1)
    print("Verified Number of Matches:",len(verified_matches))
    return verified_matches


# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
    # Get matches from the images
    verified_matches = matches_d(img1, img2)

    # Initialize SIFT
    sift = cv2.SIFT_create(1000)

    # Extract keypoints and descriptors
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # Minimum number of matches
    min_matches = 8
    if len(verified_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append(k1[match[0]].pt)
            img2_pts.append(k2[match[1]].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        print('Error: Not enough matches')
        exit()


# Equalize Histogram of Color Images
def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


def stitch_2_images(img1, img2):
    img1 = equalize_histogram_color(img1)
    img2 = equalize_histogram_color(img2)

    # Use SIFT to find keypoints and return homography matrix
    M = get_sift_homography(img1, img2)

    # Stitch the images together using homography matrix
    result_image = get_stitched_image(img2, img1, M)

    return result_image


def spatial_overlaps_matrix(imgs):
    overlap_arr = np.zeros((len(imgs), len(imgs)))
    for i in range(0, len(imgs)):
        for j in range(0, len(imgs)):
            verified_matches = matches_d(imgs[i], imgs[j])
            matches_length = len(verified_matches)
            print('matches_length', matches_length)
            if matches_length > 20:
                overlap_arr[i][j] = 1
            else:
                overlap_arr[i][j] = 0

    return overlap_arr


def stitch(imgmark, N,
           savepath=''):  # For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N + 1)]
    imgs = []
    kp = []
    des = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    result_image = stitch_2_images(imgs[0], imgs[1])
    if len(imgs) > 2:
        for k in range(2, len(imgs)):
            result_image = stitch_2_images(result_image, imgs[k])
    cv2.imwrite(f'{imgmark}_result.png', result_image)
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    plt.show()

    overlap_arr = spatial_overlaps_matrix(imgs)
    print('overlap_arr', overlap_arr)

    return overlap_arr


if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3', N=4, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
