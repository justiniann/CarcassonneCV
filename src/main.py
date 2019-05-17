import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    plt.imshow(labeled_img)
    plt.show()


def label_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img[:, :, 2]
    img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_thresh, 8, cv2.CV_32S)
    imshow_components(labels)

if __name__ == '__main__':
    img = os.path.join("..", "dat", "10.jpg")
    label_image(img)
