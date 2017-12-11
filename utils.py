import cv2
import numpy as np
from threading import current_thread


def debug(message):
    print "{}: {}" . format(current_thread().getName(), message)


def save_image(image, name, extension):
    cv2.imwrite("./images/{}{}".format(name, extension), image)


def draw3DPoints(image, points):
    image = cv2.imread('./images/{}'.format(image))
    median_dist = sum(x[2] for x in points)/len(points)
    min_dist = min(points, key=lambda x: x[2])[2]
    for point in points:
        if min_dist <= point[2] <= median_dist:
            color = (153, 255, 51)
        else:
            color = (204, 51, 0)
        cv2.circle(image, (point[0], point[1]), 5, color[::-1], 2)
    return image


def draw_matches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    # Also return the image if you'd like a copy
    return out
