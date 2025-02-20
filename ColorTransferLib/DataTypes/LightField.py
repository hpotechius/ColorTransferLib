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
    def __init__(self, file_path=None, size=None):
        self.__grid_size = size
        self.__image_array = self.__read(file_path)
        self.__image_height = self.__image_array[0][0].get_height()
        self.__image_width = self.__image_array[0][0].get_width()
        self.__type = "LightField"

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, file_path):
        # VideoWriter-Objekt erstellen
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path + ".avi", fourcc, 30.0, (self.__image_width, self.__image_height))

        # Schreibe jedes Bild in die Video-Datei
        for row in self.__image_array:
            for frame in row:
                # Konvertiere das Frame von RGB zu BGR
                frame_bgr = cv2.cvtColor((frame.get_raw() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

        # VideoWriter-Objekt freigeben
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
        # VideoCapture-Objekt erstellen
        cap = cv2.VideoCapture(file_path)

        # Überprüfen, ob das Video geöffnet werden konnte
        if not cap.isOpened():
            print(f"Error while opening the video file: {file_path}")
            return

        rows, cols = self.__grid_size
        frame_arrays = [[None for _ in range(cols)] for _ in range(rows)]
        frame_count = 0

        while True:
            # Frame für Frame lesen
            ret, frame = cap.read()

            # Wenn das Video zu Ende ist, brechen wir die Schleife ab
            if not ret:
                break

           # Konvertiere das Frame von BGR zu RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # Berechne die Position im 2D-Array
            row = frame_count // cols
            col = frame_count % cols

            if row < rows:
                frame_arrays[row][col] = Image(array=np.array(frame_rgb, dtype=np.float32) / 255.0, normalized=True)
                frame_count += 1
            else:
                break

        # VideoCapture-Objekt freigeben
        cap.release()

        # Konvertiere die Liste der Listen in ein NumPy-Array
        #frame_arrays = np.array(frame_arrays)


        return frame_arrays