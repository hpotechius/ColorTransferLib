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
from plyfile import PlyData, PlyElement

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

        self._file_extension = file_path.split('.')[-1]

        # Initialisiere leere NumPy-Arrays
        self.__colors = np.empty((0, 4), dtype=np.float32)  # Leeres Array für Farben (4 Kanäle)
        self.__positions = np.empty((0, 3), dtype=np.float32)  # Leeres Array für Positionen (3 Koordinaten)
        self.__scales = np.empty((0, 3), dtype=np.float32)  # Leeres Array für Skalen (3 Werte)
        self.__rotations = np.empty((0, 4), dtype=np.float32)  # Leeres Array für Rotationen (4 Werte)
        self.__normals = np.empty((0, 3), dtype=np.float32)  # Leeres Array für normalen (3 Werte)
        self.__params = np.empty((0, 48), dtype=np.float32)  # Leeres Array für Parmater der Spherical Harmonics (45 Werte)
        self.__opacity = np.empty((0, 1), dtype=np.float32)  # Leeres Array für die Druchsichtigkeit der Gaußverteilung (1 Wert)
        
        if self._file_extension == "ply":
            self.__read_ply_file(file_path)
        elif self._file_extension == "splat":
            self.__read_splat_file(file_path)
        elif self._file_extension == "ksplat":
            self.__read_ksplat_file(file_path)
        else:
            print("Unknown file extension")


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def __write_ply_file(self, ply_file_path):
        vertex_data = np.zeros(len(self.__positions), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            *[(f'f_dc_{i}', 'f4') for i in range(3)],
            *[(f'f_rest_{i}', 'f4') for i in range(45)]
        ])

        vertex_data['x'] = self.__positions[:, 0]
        vertex_data['y'] = self.__positions[:, 1]
        vertex_data['z'] = self.__positions[:, 2]
        vertex_data['nx'] = self.__normals[:, 0]
        vertex_data['ny'] = self.__normals[:, 1]
        vertex_data['nz'] = self.__normals[:, 2]
        vertex_data['opacity'] = self.__opacity[:, 0]
        vertex_data['scale_0'] = self.__scales[:, 0]
        vertex_data['scale_1'] = self.__scales[:, 1]
        vertex_data['scale_2'] = self.__scales[:, 2]
        vertex_data['rot_0'] = self.__rotations[:, 0]
        vertex_data['rot_1'] = self.__rotations[:, 1]
        vertex_data['rot_2'] = self.__rotations[:, 2]
        vertex_data['rot_3'] = self.__rotations[:, 3]
        for i in range(3):
            vertex_data[f'f_dc_{i}'] = self.__params[:, i]
        for i in range(45):
            vertex_data[f'f_rest_{i}'] = self.__params[:, i + 3]

        el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([el]).write(ply_file_path)
    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, output_file_path, extension="splat"):
        if extension == "ply" and self._file_extension == "ply":
            self.__write_ply_file(output_file_path + ".ply")
        elif extension == "splat" and self._file_extension == "splat":
            with open(output_file_path + "." + extension, 'wb') as f:
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
        else:
            print("The file extension does not match the original file type")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # def get_splat_data(self):
    #     return self.__splat_data
    
    def get_type(self):
        return self.__type
    
    def get_colors(self):
        # remove alpha channel
        colors_wo_alpha = self.__colors[:, :3]
        # reshape to (len(colors), 1, 3)
        return colors_wo_alpha.reshape(-1, 1, 3)
        # return self.__colors

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
        # reshape to (len(colors), 3)
        reshaped_cols = colors.reshape(-1, 3)

        # Extract the fourth channel from self.__colors
        fourth_channel = self.__colors[:, 3].reshape(-1, 1)

        # Concatenate reshaped_cols with the fourth channel
        reshaped_cols_with_alpha = np.concatenate((reshaped_cols, fourth_channel), axis=1)

        self.__colors = reshaped_cols_with_alpha

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def __read_ply_file(self, ply_file_path):
        ply_data = PlyData.read(ply_file_path)
        vertex_data = ply_data['vertex'].data

        self.__positions = np.vstack((self.__positions, np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T))
        self.__normals = np.vstack((self.__normals, np.vstack((vertex_data['nx'], vertex_data['ny'], vertex_data['nz'], np.zeros(len(vertex_data)))).T))
        self.__params = np.vstack((self.__params, np.vstack([vertex_data[f'f_dc_{i}'] for i in range(3)] + [vertex_data[f'f_rest_{i}'] for i in range(45)]).T))
        self.__opacity = np.vstack((self.__opacity, vertex_data['opacity'].reshape(-1, 1)))
        self.__scales = np.vstack((self.__scales, np.vstack((vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2'])).T))
        self.__rotations = np.vstack((self.__rotations, np.vstack((vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3'])).T))

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