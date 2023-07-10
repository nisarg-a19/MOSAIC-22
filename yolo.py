from helpers import load_weights, filter_boxes, non_max_suppress, generate_colors, draw_boxes, download_weights, add_text
from architecture import create_model, LABELS, IMAGE_H, IMAGE_W

import numpy as np
from helpers import model
from tqdm import tqdm
import cv2
import sys
import os
import argparse
import struct
import numpy as np
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.models import load_model

# Thresholds for confidence score and non-max suppression
OBJ_THRESHOLD = 0.6
NMS_THRESHOLD = 0.5

model = load_model('model2.h5')
# All yolo actions from input to output
def make_yolo(original_image, model):

    input_image = cv2.resize(original_image, (IMAGE_H, IMAGE_W)) / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    yolo_output = np.squeeze(model.predict(input_image))
    boxes = filter_boxes(yolo_output, OBJ_THRESHOLD)
    boxes = non_max_suppress(boxes, NMS_THRESHOLD)
    colours = generate_colors(LABELS)
    output_image = draw_boxes(original_image, boxes, LABELS, colours)
 
    return output_image



################### TEST YOLO ON IMAGE ###################

# Objects detection from image
def yolo_image(image_path, model):

    original_image = cv2.imread(image_path)
    image = make_yolo(original_image, model)
    new_path = 'images/yolo_' + image_path.split('/')[-1]
    cv2.imwrite(new_path, image)
    print("Output file saved to:", new_path)
    cv2.imshow('image',image)
    k = cv2.waitKey(0)
    if k == 'q':         # wait for ESC key to exit
        cv2.destroyAllWindows()


################### TEST YOLO ON VIDEO ###################

# Objects detection from video
def yolo_video(video_path, model, faster_times=1):

    # Path for output video
    video_out = 'images/yolo_' + video_path.split('/')[-1]

    # Set video reader and writer
    video_reader = cv2.VideoCapture(video_path)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'),
                                   fps * faster_times, (frame_w, frame_h))
    # Iterate over all frames
    for _ in tqdm(range(nb_frames)):

        ret, original_image = video_reader.read()
        image = make_yolo(original_image, model)
        video_writer.write(np.uint8(image))

    video_reader.release()
    video_writer.release()
    print("Output file saved to:", video_out)



################### YOLO LIVE STREAM ###################

# Objects detection from live stream of webcam
def yolo_live(model, mirror=True):

    cap = cv2.VideoCapture(0)

    # Capture frame-by-frame until quit
    while(True):

        ret, frame = cap.read()
        if mirror:
            frame = cv2.flip(frame, 1)
        image = make_yolo(frame, model)
        image = add_text(image)

        # Display the resulting frame
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

model = load_model('model2.h5')
load_weights(model,'yolov3_custom_final.weights')
yolo_image("images/52.PNG", model)

# yolo = create_model()
# load_weights(yolo, 'yolov3_custom_1000.weights')
    

#yolo_image("1.PNG",create_model())

# Run yolo
if __name__ == '__main__':
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description='yolo v2')

    # Add arguments
    parser.add_argument('action', type=str, default='', help='A required integer positional argument')
    parser.add_argument('path_to_file', type=str, default='', nargs='?', help='/path/to/file')

    # Parse arguments
    args = parser.parse_args()

    # Parse arguments
    action = args.action
    path_to_file = args.path_to_file

    if action not in ['download_weights', 'run_picture', 'run_video', 'run_live']:
        print('Please specify the action: [download_weights, run_picture, run_video, run_live]')
        sys.exit()

    if action == 'download_weights':
        download_weights()

    if action == 'run_picture':
        if os.path.isfile(path_to_file):
            # Create model and load weights
            yolo = create_model()
            load_weights(yolo, 'yolov3_custom_final.weights')
            yolo_image(path_to_file, yolo)
        else:
            print('Enter valid path to file.')
            sys.exit()

    if action == 'run_video':
        if os.path.isfile(path_to_file):
            # Create model and load weights
            yolo = create_model()
            load_weights(yolo, 'yolov3_custom_final.weights')
            yolo_video(path_to_file, yolo, faster_times=1)
        else:
            print('Enter valid path to file.')
            sys.exit()

    if action == 'run_live':
        # Create model and load weights
        yolo = create_model()
        load_weights(yolo, 'yolov3_custom_final.weights')
        yolo_live(yolo)