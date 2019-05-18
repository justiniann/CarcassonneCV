import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_image(img):
    plt.imshow(img)
    plt.show()


def imshow_components(labels):
    current_num = 0
    size_threshold = labels.size * 0.001
    corrected_labels = labels.copy()
    for l in np.unique(labels):
        if l == 0:
            continue
        new_label = 0
        l_pixels = np.where(labels == l)
        if len(l_pixels[0]) > size_threshold:
            current_num += 1
            new_label = current_num
        for x, y in zip(l_pixels[0], l_pixels[1]):
            corrected_labels[x, y] = new_label

    # Map component labels to hue val
    label_hue = np.uint8(179*corrected_labels/np.max(corrected_labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img


def label_image(img_path, sensitivity=30):
    # load image in HSV
    img = cv2.imread(img_path)
    background = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of colors that are indicative of fields
    lower_green = np.array([60 - sensitivity, 100, 100])
    upper_green = np.array([60 + sensitivity, 255, 255])

    # create a mask, to isolate the fields from everything else on the board
    only_show_fields = cv2.inRange(hsv_img, lower_green, upper_green)

    # use the mask to turn all field pixels to white, and everything else to black
    res = cv2.bitwise_and(img, img, mask=only_show_fields)

    # threshold the image, so that the fields are white and everything else is black
    res_thresholded = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY)[1]

    # dilate and erode, to lessen the effect of slightly disconnected tiles
    res_thresholded = cv2.dilate(res_thresholded, np.ones((2, 2), np.uint8), iterations=1)
    res_thresholded = cv2.erode(res_thresholded, np.ones((3, 3), np.uint8), iterations=1)
    # show_image(res_thresholded)

    # convert from HSV to grayscale
    res_thresholded = cv2.cvtColor(res_thresholded, cv2.COLOR_HSV2BGR)
    res_thresholded = cv2.cvtColor(res_thresholded, cv2.COLOR_BGR2GRAY)

    # run connected component analysis on the thresholded image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(res_thresholded, connectivity=8, ltype=cv2.CV_32S)
    foreground = imshow_components(labels)
    show_image(foreground)

    # mask the fields in the original image, so that we can overlay our color coded fields
    dont_show_fields = cv2.bitwise_not(only_show_fields)
    background = cv2.bitwise_and(background, background, mask=dont_show_fields)
    # show_image(background)

    final = cv2.bitwise_or(foreground, background)
    show_image(final)


if __name__ == '__main__':
    label_image(os.path.join("..", "dat", "example.jpg"))
