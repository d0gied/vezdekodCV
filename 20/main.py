import cv2 as cv
import csv
import typing
import os
import torch

# or yolov5n - yolov5x6, custom

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
    
def find_cars(input_dir: str, output_cars: str = 'output.csv'): 
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')

    input_folder = os.path.normpath(input_dir)
    images = merge_channels(input_folder, 'images')

    print('Looking for cars...')
    frames = []
    for image in images:
        print(f'Process {image}')
        results = model(image).pandas().xyxy[0]
        results = results[(results['name'] == 'car') | (results['name'] == 'bus') | (results['name'] == 'truck')]
        cars = []
        if results.empty:
            continue
        for index, row in results.iterrows():
            x, y = row['xmin'], row['ymin']
            w, h = row['xmax'] - x, row['ymax'] - y
            cars.append((x, y, w, h))
        frames.append(cars)

    with open(output_cars, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        for file, result in zip(images, frames):
            writer.writerow([file, bool(result)])

def validate(val_file: str='val.csv', output_file: str ='output.csv'):
    val = csv.reader(open(val_file, newline='\n'))
    current = csv.reader(open(output_file, newline='\n'))
    cnt = 0
    index = 1
    for cur_data, val_data in zip(current, val):
        if cur_data[1] != val_data[1]:
            print(index, "Expected:", val_data[1], "Received:", cur_data[1]) 
        else:
            cnt += 1
        index += 1

    print(f'{cnt / index * 100}%')

if __name__ == "__main__":
    find_cars('dataset')
    # validate()

    