# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications by Surya Vasudev


"""Main script to run the object detection routine."""
import argparse
import sys

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

#App window

from tkinter import *
from PIL import Image, ImageTk
import webbrowser


def querycheck():
  if detection_result.detections[0].categories[0].category_name == 'person':
    webbrowser.open(f'https://duckduckgo.com/?q={detection_result.detections[0].categories[0].category_name}')
  else:
    webbrowser.open(f'https://shopping.google.com/search?tbm=shop&hl=en-US&psb=1&ved=2ahUKEwiQ7pDCnNz8AhXcC4gJHc6LB0cQu-kFegQIABAK&q={detection_result.detections[0].categories[0].category_name}')


# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("1920x720")

# Create a Label to capture the Video frames
label =Label(win)
label.grid(row=0, column=0)

#button

btn = Button(win, text = 'Search Detected', bd = '5',command = lambda: querycheck())
btn.grid(row=10, column = 10)     



def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """


  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  #Taken out of the while loop to make sure that global imagecv is not called each time
  global imagecv
  success, imagecv = cap.read()
  if not success:
    sys.exit(
      'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    )

  # Continuously capture images from the camera and run inference
  while cap.isOpened():

    success, imagecv = cap.read()

    imagecv = cv2.flip(imagecv, 1)

    # Convert the imagecv from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(imagecv, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB imagecv.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    global detection_result
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input imagecv
    imagecv = utils.visualize(imagecv, detection_result)
    
    #Fix output image colorspace for tkinter
    rgb_image = cv2.cvtColor(imagecv, cv2.COLOR_BGR2RGB)
    
    #Update tkinter window
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_image))
    label.configure(image=imgtk)
    win.update_idletasks()
    win.update()



def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
