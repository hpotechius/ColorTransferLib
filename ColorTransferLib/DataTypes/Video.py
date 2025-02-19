"""
Copyright 2024 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import os
import numpy as np
import subprocess

from ColorTransferLib.DataTypes.Image import Image

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Video:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, file_path=None, imgs=None, fps=0.0):
        self.__type = "Video"
        self.__fps = fps

        if file_path is not None:
            frames = self.split_video_into_frames(file_path)
            self.__imgs = [Image(array=frame) for frame in frames]
        elif imgs is not None:
            self.__imgs = imgs

    # Function to split video into frames
    def split_video_into_frames(self, video_path):
        # Capture the video from the file
        cap = cv2.VideoCapture(video_path) 
        
        # Get the framerate of the video
        self.__fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        frames = []

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If the frame was read successfully, save it as an image
            if ret:
                frames.append(frame)
                frame_count += 1
            else:
                break
        # Release the video capture object
        cap.release()
        return frames
    
    
    # ------------------------------------------------------------------------------------------------------------------
    # Writes the file to the specified path.
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, out_path):
        height, width = self.__imgs[0].get_width(), self.__imgs[0].get_height()
        size = (height, width)
        
        #Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        # Initialize the VideoWriter
        avi_path = out_path + ".avi"
        out = cv2.VideoWriter(out_path + ".avi", fourcc, self.__fps, size)

        print(self.__fps)

        for i, frame in enumerate(self.__imgs):
            ff = (cv2.cvtColor(frame.get_raw(), cv2.COLOR_RGB2BGR) * 255.0).astype(np.uint8)

            if isinstance(ff, np.ndarray):
                out.write(ff)
            else:
                print(f"Frame {i} is not a valid numpy array and will be skipped.")
        
        out.release()

        mp4_path = out_path + ".mp4"
        subprocess.run(['ffmpeg', '-i', avi_path, '-vcodec', 'libx264', '-acodec', 'aac', mp4_path])


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
        
    def get_fps(self):
        return self.__fps

    def get_type(self):
        return self.__type

    # ------------------------------------------------------------------------------------------------------------------
    # return the color values as vector
    # ------------------------------------------------------------------------------------------------------------------
    def get_colors(self):
        return [img.get_colors() for img in self.__imgs]

    # ------------------------------------------------------------------------------------------------------------------
    # returns the image as numpy array
    # ------------------------------------------------------------------------------------------------------------------
    def get_raw(self):
        return [img.get_raw() for img in self.__imgs]

    # ------------------------------------------------------------------------------------------------------------------
    # returns the image 
    # ------------------------------------------------------------------------------------------------------------------
    def get_images(self):
        return self.__imgs