"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""
import cv2
import numpy as np
from ColorTransferLib.DataTypes.Image import Image
import subprocess
import os
import re


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class LightField:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # The file has to be a .mp4 file. And the grid size of the light field has to be given as a tuple (rows, cols).
    # In case of a 10x10 grid with images of size 256x256x3, the resulting image_array will have the shape 
    # (10, 10, 256, 256, 3).
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, file_path=None, isVideo=True, size=None):
        self.__grid_size = size
        self.__isVideo = isVideo
        self.__image_array = self.__read(file_path)

        first_frame = next((frame for row in self.__image_array for frame in row if frame is not None), None)
        if first_frame is None:
            raise ValueError(f"No frames could be loaded from path: {file_path}")

        self.__image_height = first_frame.get_height()
        self.__image_width = first_frame.get_width()
        self.__type = "LightField"

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, file_path):
        if not self.__isVideo:
            # create output folder named after last segment of file_path
            folder_name = os.path.basename(file_path.rstrip('/'))
            os.makedirs(file_path, exist_ok=True)

            idx = 1
            pad = 4
            for row in self.__image_array:
                for frame in row:
                    if frame is None:
                        continue
                    filename = f"{folder_name}_{idx:0{pad}d}.png"
                    out_path = os.path.join(file_path, filename)
                    img = (frame.get_raw() * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_path, img_bgr)
                    idx += 1
        else:
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(file_path + ".avi", fourcc, 30.0, (self.__image_width, self.__image_height))

            # Write each image to the video file
            for row in self.__image_array:
                for frame in row:
                    # Convert the frame from RGB to BGR
                    frame_bgr = cv2.cvtColor((frame.get_raw() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

            # Release VideoWriter object
            out.release()
            
            avi_path = file_path + ".avi"
            mp4_path = file_path + ".mp4"

            subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-vcodec', 'libx264', '-acodec', 'aac', mp4_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Delete the temporary AVI file
            if os.path.exists(avi_path):
                os.remove(avi_path)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_image_array(self):
        return self.__image_array
    
    def get_grid_size(self):
        return self.__grid_size
    
    def get_type(self):
        return self.__type

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # SETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_image_array(self, image_array):
        self.__image_array = image_array

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __read(self, file_path):
        rows, cols = self.__grid_size
        frame_arrays = [[None for _ in range(cols)] for _ in range(rows)]
        frame_count = 0

        if not self.__isVideo and os.path.isdir(file_path):
            # If isVideo=True and file_path is a directory, load images in natural order.

            valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

            def natural_key(name):
                return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

            image_files = [
                name for name in os.listdir(file_path)
                if os.path.isfile(os.path.join(file_path, name)) and os.path.splitext(name)[1].lower() in valid_exts
            ]
            image_files = sorted(image_files, key=natural_key)

            for image_name in image_files:
                image_path = os.path.join(file_path, image_name)
                frame = cv2.imread(image_path)

                if frame is None:
                    continue

                # Convert the frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate the position in the 2D array
                row = frame_count // cols
                col = frame_count % cols

                if row < rows:
                    frame_arrays[row][col] = Image(array=np.array(frame_rgb, dtype=np.float32) / 255.0, normalized=True)
                    frame_count += 1
                else:
                    break
        else:
            if not os.path.isfile(file_path):
                print(f"Error: The given path is not a valid video file: {file_path}")
                return frame_arrays

            # Create VideoCapture object
            cap = cv2.VideoCapture(file_path)

            # Check whether the video file could be opened
            if not cap.isOpened():
                print(f"Error while opening the video file: {file_path}")
                return frame_arrays

            while True:
                # Read frame by frame
                ret, frame = cap.read()

                # Stop when the video has ended
                if not ret:
                    break

                # Convert the frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate the position in the 2D array
                row = frame_count // cols
                col = frame_count % cols

                if row < rows:
                    frame_arrays[row][col] = Image(array=np.array(frame_rgb, dtype=np.float32) / 255.0, normalized=True)
                    frame_count += 1
                else:
                    break

            # Release VideoCapture object
            cap.release()

        # Convert the list of lists to a NumPy array
        #frame_arrays = np.array(frame_arrays)


        return frame_arrays