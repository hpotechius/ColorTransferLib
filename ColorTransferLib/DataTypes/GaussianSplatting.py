"""
Copyright 2024 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class GaussianSplatting:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, file_path=None):
        self.__type = "GaussianSplatting"
        print("GaussianSplatting: ", file_path)

        # Initialisiere leere NumPy-Arrays
        self.__colors = np.empty((0, 4), dtype=np.float32)  # Leeres Array für Farben (4 Kanäle)
        self.__positions = np.empty((0, 3), dtype=np.float32)  # Leeres Array für Positionen (3 Koordinaten)
        self.__scales = np.empty((0, 3), dtype=np.float32)  # Leeres Array für Skalen (3 Werte)
        self.__rotations = np.empty((0, 4), dtype=np.float32)  # Leeres Array für Rotationen (4 Werte)
        
        self.__read_splat_file(file_path)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, output_file_path, extension=".splat"):
        with open(output_file_path + extension, 'wb') as f:
            for i in range(len(self.__positions)):
                # Schreibe die Position (3 float32)
                f.write(self.__positions[i].astype(np.float32).tobytes())

                # Schreibe die Skalen (3 float32)
                f.write(self.__scales[i].astype(np.float32).tobytes())

                # Schreibe die Farbe (4 uint8)
                f.write((self.__colors[i] * 255).astype(np.uint8).tobytes())

                # Normiere die Rotationswerte in den Bereich [0, 255] und schreibe sie (4 uint8)
                rot_normalized = ((self.__rotations[i] * 128) + 128).clip(0, 255).astype(np.uint8)
                f.write(rot_normalized.tobytes())

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_splat_data(self):
        return self.__splat_data
    
    def get_type(self):
        return self.__type
    
    def get_colors(self):
        return self.__colors

    def get_positions(self):
        return self.__positions
    
    def get_scales(self):
        return self.__scales
    
    def get_rotations(self):
        return self.__rotations

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # SETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_colors(self, colors):
        self.__colors = colors

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def __read_splat_file(self, splat_file_path):
        with open(splat_file_path, 'rb') as f:
            data = f.read()

        # Der Puffer wird Byte für Byte gelesen, abhängig von den Datentypen
        offset = 0

        # Initialisiere leere Listen für die einzelnen Arrays
        positions = []
        scales = []
        colors = []
        rotations = []

        while offset < len(data):
            # Lese die Position (3 float32)
            position = np.frombuffer(data, dtype=np.float32, count=3, offset=offset)
            offset += 3 * 4  # 3 Werte * 4 Bytes pro float32

            # Lese die Skalen (3 float32)
            scale = np.frombuffer(data, dtype=np.float32, count=3, offset=offset)
            offset += 3 * 4

            # Lese die Farbe (4 uint8)
            color = np.frombuffer(data, dtype=np.uint8, count=4, offset=offset) 
            offset += 4

            # transfer color to float32 and normalize
            color = color.astype(np.float32) / 255

            # Lese die Rotation (4 uint8)
            rot = np.frombuffer(data, dtype=np.uint8, count=4, offset=offset)
            offset += 4

            # Normiere die Rotationswerte zurück auf den ursprünglichen Bereich [-1, 1]
            rot_normalized = (rot.astype(np.float32) - 128) / 128

            # Füge die gelesenen Werte zu den Listen hinzu
            positions.append(position)
            scales.append(scale)
            colors.append(color)
            rotations.append(rot_normalized)

        # Konvertiere die Listen in NumPy-Arrays
        self.__positions = np.array(positions)
        self.__scales = np.array(scales)
        self.__colors = np.array(colors)
        self.__rotations = np.array(rotations)