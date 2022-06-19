import cv2 as cv
import csv
import typing
import os
import torch
from torch import nn
import numpy as np


class ConvNet(nn.Module):
    def __init__(self):
         super(ConvNet, self).__init__()
         self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8),
            nn.ReLU(), nn.MaxPool2d(kernel_size=8))
         self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(), nn.MaxPool2d(kernel_size=4))
         self.drop_out = nn.Dropout()
         self.fc1 = nn.Linear(576, 128)
         self.fc2 = nn.Linear(128, 10)

    def forward(self, x): 
      out = self.layer1(x) 
      out = self.layer2(out) 
      out = out.reshape(out.size(0), -1) 
      out = self.drop_out(out) 
      out = self.fc1(out) 
      out = self.fc2(out) 
      return out

# or yolov5n - yolov5x6, custom
cmodel = torch.hub.load('ultralytics/yolov5', 'yolov5m')
my_model = torch.load('model_v3.pt')

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
    
def get_car(image):
    result = cmodel(image).pandas().xyxy[0] # use yolo to find vehicles 
    cars = result[(result['name'] == 'car') | (result['name'] == 'bus') | (result['name'] == 'truck')][:]
    if not cars.empty:
        cars['criteria'] = (cars['xmax'] - cars['xmin']) * (cars['ymax'] - cars['ymin']) # chouse rect with biggest area
        car = cars.loc[cars['criteria'].idxmax()]
        x, y = int(car['xmin']), int(car['ymin'])
        w, h = int(car['xmax']) - x, int(car['ymax']) - y
        return (x, y, w, h)
    else:
        return None

def prepare_image(image): # preparation image for model
    img = cv.imread(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return (cv.resize(img, (128, 128)) / 255)


def find_color(input_dir, output_file="output_color.csv"):
    input_folder = os.path.normpath(input_dir)
    images = merge_channels(input_folder, 'images')

    target_colors = [
    'black',
    'blue',
    'green',
    'grey',
    'orange',
    'red',
    'silver',
    'white',
    'yellow',
    ]
    colors = { # links bitween model and answer colors
        'black': 'black',
        'blue': 'blue_cyan',
        'green': 'green',
        'grey': 'black',
        'orange': 'red',
        'red': 'red',
        'silver': 'white_silver',
        'white': "white_silver",
        'yellow': 'yellow',
    }

    results = []
    for image in images:
        print(f'Cropping {image}')
        x, y, w, h = get_car(image) # get bounding rect of the car
        img = cv.imread(image)[y:y+h, x:x+w] # crop image
        cv.imwrite('cropped.jpg', img) # save to temp file

        print(f'Calculating color {image}')

        frame = prepare_image('cropped.jpg') # prepare image for model

        with torch.no_grad():
            logps = my_model(torch.Tensor(
                np.array([frame])).permute(0, 3, 1, 2))
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        result = colors[target_colors[probab.index(max(probab))]]


        results.append(result)

    print('Saving data')
    with open(output_file, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        for file, result in zip(images, results):
            writer.writerow([file, result])



def validate(val_file: str='colors.csv', output_file: str ='output_color.csv'):
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
    find_color('color_dataset')
    validate()


    