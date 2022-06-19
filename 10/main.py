import cv2 as cv
import csv
import typing
import os

def get_amount(folder: str) -> int:  # returns number of images from file
    return int(open(f'{folder}\\image_counter.txt', 'r').read())


def parse_description(folder: str) -> typing.List[typing.Dict]: 
    amount = get_amount(folder)
    images = [{'r': '', 'g': '', 'b': ''} for i in range(amount)]
    with open(f'{folder}\\description.csv', newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in list(reader)[:amount * 3]:
            images[int(line['full_image_index']) - 1][line['color']] = f'{folder}\\data\\{line["image_path"]}'
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
    

if __name__ == "__main__":
    images = merge_channels('dataset', 'images')
