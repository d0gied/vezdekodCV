import cv2 as cv
import csv
import typing
import os
import torch
import numpy as np
import colorsys
from matplotlib import pyplot as plt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from sklearn.cluster import KMeans
import utils
# or yolov5n - yolov5x6, custom


def get_amount(folder: str) -> int:  # returns number of images from file
    return int(open(f'{folder}\\image_counter.txt', 'r').read())


def parse_description(folder: str) -> typing.List[typing.Dict]:
    amount = get_amount(folder)
    images = [{'r': '', 'g': '', 'b': ''} for i in range(amount)]
    with open(f'{folder}\\description.csv', newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in list(reader)[:amount * 3]:
            images[int(line['full_image_index']) - 1][line['color']
                                                      ] = f'{folder}\\data\\{line["image_path"]}'
    return images


def merge_channels(input_dir: str, output_dir: str) -> None:
    input_folder = os.path.normpath(input_dir)
    output_folder = os.path.normpath(output_dir)
    images_description = parse_description(input_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    images = []
    for index, image_data in enumerate(images_description):
        r_img = cv.imread(image_data['r'], cv.IMREAD_GRAYSCALE)
        g_img = cv.imread(image_data['g'], cv.IMREAD_GRAYSCALE)
        b_img = cv.imread(image_data['b'], cv.IMREAD_GRAYSCALE)
        image = cv.merge([b_img, g_img, r_img])

        output_path = f'{output_folder}\\{index + 1:05d}.jpg'
        cv.imwrite(output_path, image)
        print(f'Prepared: {output_path}')
        images.append(output_path)
    return images


def L(color1):
    color1_rgb = sRGBColor(*(color1 / 255))
    color1_lab = convert_color(color1_rgb, LabColor)
    return color1_rgb.VALUES

def get_color(img):
    height, width = img.shape[:2]
    k = 400 / height
    image = cv.resize(img, (int(width * k), 400)) # resize image, as if it makes code faster
    # reshape the image tobe a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # cluster the pixel intensities
    clt = KMeans(n_clusters = 5)
    clt.fit(image)
    hist = utils.centroid_histogram(clt)
    colors = clt.cluster_centers_[::]
    return max(zip(colors, hist), key=lambda x: x[1])[0][::-1]


def calc_metric(image: str, x: int = 0, y: int = 0, w: int = None, h: int = None) -> typing.Tuple[int, int, int]:
    """find dominating color in region

    Args:
        image (str): path to image
        x (int): x
        y (int): y
        w (int): w
        h (int): h

    Returns:
        typing.Tuple[int, int, int]: (r, g, b)
    """
    img = cv.imread(image)

    height, width = img.shape[:2]
    if w is None:
        w = width
    if h is None:
        h = height

    img = cv.imread(image)[y:y+h, x:x+w]
    cv.imwrite('cropped.jpg', img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return get_color(img)


def visualize_metric(image: str, x: int = 0, y: int = 0, w: int = None, h: int = None):
    """vizualize dominating color in region

    Args:
        image (str): path to image
        x (int): x
        y (int): y
        w (int): w
        h (int): h

    Returns:
        typing.Tuple[int, int, int]: (r, g, b)
    """

    color = calc_metric(image, x, y, w, h)
    img = cv.imread('cropped.jpg')
    height, width = img.shape[:2]
    rect = np.zeros((height, width // 3, 3), np.uint8)
    rect[:] = color

    vis = np.concatenate((img, rect), axis=1)
    vis = cv.cvtColor(vis, cv.COLOR_BGR2RGB)
    plt.imshow(vis)
    plt.show()


if __name__ == "__main__":
    print(calc_metric('images\\00071.jpg', 130, 300, 800, 300))
    visualize_metric(f'images\\00071.jpg', 130, 300, 800, 300)


