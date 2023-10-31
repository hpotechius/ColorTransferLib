import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
#os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
os.environ["OCTAVE_EXECUTABLE"] = "/opt/homebrew/bin/octave-cli"
from oct2py import octave, Oct2Py
from scipy.interpolate import interp1d


class HistogramMatching():
   # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def histogram_matching2(src, ref, iterations):

        # Preprocessing
        src_color = src
        ref_color = ref
        out_img = copy.deepcopy(src)

        # [1] Change range from [0.0, 1.0] to [0, 255] and copy source and reference to GPU and create output
        device_src = src_color.squeeze()
        device_ref = ref_color.squeeze()

        for t in range(iterations):
            print(t)
            sci_mat = R.random()#random_state=5)
            mat_rot = sci_mat.as_matrix()

            mat_rot_inv = sci_mat.inv().as_matrix()

             # [2] Create random 3x3 rotation matrix
            mat_rot_tile = np.tile(mat_rot,(src_color.shape[0], 1, 1))
            mat_rot_inv_tile = np.tile(mat_rot_inv,(src_color.shape[0], 1, 1))

            #print(device_ref[0])
            # [3] Rotate source and reference colors with random rotation matrix
            src_rotated = np.einsum('ilk,ik->il', mat_rot_tile, device_src)
            ref_rotated = np.einsum('ilk,ik->il', mat_rot_tile, device_ref)


            eps = 1e-6

            src_rotated_temp = copy.deepcopy(src_rotated)
            for i in range(3):
                inp_src = src_rotated[:,i]
                inp_ref = ref_rotated[:,i]

                datamin = np.min([inp_src, inp_ref]) - eps
                datamax = np.max([inp_src, inp_ref]) + eps
                u = np.linspace(datamin, datamax, 300)

                # Compute the histograms for each color channel
                hist_src, _ = np.histogram(inp_src.flatten(), u)
                hist_ref, _ = np.histogram(inp_ref.flatten(), u)

                #  Compute the CDFs for each color channel
                input_cdf_r = np.cumsum(hist_src + eps)
                input_cdf_r = input_cdf_r / input_cdf_r[-1]
                ref_cdf_r = np.cumsum(hist_ref + eps)
                ref_cdf_r = ref_cdf_r / ref_cdf_r[-1]

                # Compute the mapping function for each color channel
                mapping = np.interp(input_cdf_r, ref_cdf_r, range(299))

                f_interp = interp1d(u[:-1], mapping, kind='linear', bounds_error=False, fill_value=(mapping[0], mapping[-1]))

                inp_interp = f_interp(inp_src)
                src_rotated_temp[:,i] = (inp_interp - 1) / (300 - 1) * (datamax - datamin) + datamin

            # [7] Rotate Back
            # relaxation = 1.0
            # shift = src_rotated_temp - src_rotated
            # device_src = np.einsum('ilk,ik->il', mat_rot_inv_tile, shift)
            # device_src = relaxation * device_src + src_rotated
            device_src = np.einsum('ilk,ik->il', mat_rot_inv_tile, src_rotated_temp)
        
        device_src = np.clip(device_src, 0, 1)

        return device_src.astype("float32")
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def histogram_matching(src_color, ref_color, iterations):
        # [1] Change range from [0.0, 1.0] to [0, 255] and copy source and reference to GPU and create output
        device_src = copy.deepcopy(src_color)
        device_ref = copy.deepcopy(ref_color)
        # device_src[:2] = src_color[:2] + 255.0
        # device_ref[:2] = ref_color[:2] + 255.0

        m = 1.0
        soft_m = 1.0 / m
        max_range = 900
        stretch = round(math.pow(max_range, soft_m))
        c_range = int(stretch * 2 + 1)

        for t in range(iterations):
            #print(t)
            sci_mat = R.random()#random_state=5)
            mat_rot = sci_mat.as_matrix()
            mat_rot_inv = sci_mat.inv().as_matrix()

             # [2] Create random 3x3 rotation matrix
            mat_rot_tile = np.tile(mat_rot,(src_color.shape[0], 1, 1))
            mat_rot_inv_tile = np.tile(mat_rot_inv,(src_color.shape[0], 1, 1))

            mat_rot_tile_ref = np.tile(mat_rot,(ref_color.shape[0], 1, 1))
            mat_rot_inv_tile_ref = np.tile(mat_rot_inv,(ref_color.shape[0], 1, 1))

            # [3] Rotate source and reference colors with random rotation matrix
            src_rotated = np.einsum('ikl,ik->il', mat_rot_tile, device_src)
            ref_rotated = np.einsum('ikl,ik->il', mat_rot_tile_ref, device_ref)

            # [4] Get 1D marginal
            src_marg_x = src_rotated[:,0]
            src_marg_y = src_rotated[:,1]
            src_marg_z = src_rotated[:,2]
            ref_marg_x = ref_rotated[:,0]
            ref_marg_y = ref_rotated[:,1]
            ref_marg_z = ref_rotated[:,2]

            # [5] Calculate 1D pdf for range [-255, 255] which has to be shifted to [0, 884] (without stretching) in order
            # to allow indexing. The points can be rotated into another octant, therefore the range has to be extended from
            # [0, 255] (256 color values) to [-442, 442] (885 color values). The value 442 was chosen because a color value
            # of (255, 255, 255) can be rotated to (441.7, 0, 0).
            src_cum_marg_x = np.histogram(src_marg_x, bins=c_range, range=(-max_range, max_range), density=True)[0]
            src_cum_marg_y = np.histogram(src_marg_y, bins=c_range, range=(-max_range, max_range), density=True)[0]
            src_cum_marg_z = np.histogram(src_marg_z, bins=c_range, range=(-max_range, max_range), density=True)[0]

            ref_cum_marg_x = np.histogram(ref_marg_x, bins=c_range, range=(-max_range, max_range), density=True)[0]
            ref_cum_marg_y = np.histogram(ref_marg_y, bins=c_range, range=(-max_range, max_range), density=True)[0]
            ref_cum_marg_z = np.histogram(ref_marg_z, bins=c_range, range=(-max_range, max_range), density=True)[0]


            # [6] Calculate cumulative 1D pdf
            src_cum_marg_x = np.cumsum(src_cum_marg_x)
            src_cum_marg_y = np.cumsum(src_cum_marg_y)
            src_cum_marg_z = np.cumsum(src_cum_marg_z)

            ref_cum_marg_x = np.cumsum(ref_cum_marg_x)
            ref_cum_marg_y = np.cumsum(ref_cum_marg_y)
            ref_cum_marg_z = np.cumsum(ref_cum_marg_z)


            # Create LUT
            lut_x = np.zeros(c_range)
            lut_y = np.zeros(c_range)
            lut_z = np.zeros(c_range)

            for i, elem in enumerate(src_cum_marg_x):
                absolute_val_array = np.abs(ref_cum_marg_x - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_x[int(i)] = smallest_difference_index
            for i, elem in enumerate(src_cum_marg_y):
                absolute_val_array = np.abs(ref_cum_marg_y - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_y[int(i)] = smallest_difference_index
            for i, elem in enumerate(src_cum_marg_z):
                absolute_val_array = np.abs(ref_cum_marg_z - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_z[int(i)] = smallest_difference_index

            
            # Adapt src values
            transferred_rotated_x = lut_x[np.clip(src_marg_x.astype("int64") + stretch, 0, c_range-1)]
            transferred_rotated_y = lut_y[np.clip(src_marg_y.astype("int64") + stretch, 0, c_range-1)]
            transferred_rotated_z = lut_z[np.clip(src_marg_z.astype("int64") + stretch, 0, c_range-1)]
            # transferred_rotated_x = lut_x[src_marg_x.astype("int64") + stretch]
            # transferred_rotated_y = lut_y[src_marg_y.astype("int64") + stretch]
            # transferred_rotated_z = lut_z[src_marg_z.astype("int64") + stretch]
            transferred_rotated = np.concatenate((transferred_rotated_x[:,np.newaxis], transferred_rotated_y[:,np.newaxis]), axis=1)
            transferred_rotated = np.concatenate((transferred_rotated, transferred_rotated_z[:,np.newaxis]), axis=1)

            # [7] Rotate Back
            #transferred_rotated = np.power(transferred_rotated, 1 / soft_m) - stretch
            output = np.einsum('ikl,ik->il', mat_rot_inv_tile, transferred_rotated - stretch)

            # dist_x = np.linalg.norm(transferred_rotated_x - src_rotated[:,0])
            # dist_y = np.linalg.norm(transferred_rotated_y - src_rotated[:,1])
            # dist_z = np.linalg.norm(transferred_rotated_z - src_rotated[:,2])
            # dist = [dist_x, dist_y, dist_z]
            # print(dist)

            #output[:2] = output[:2] - 255
            device_src = np.clip(output, -255, 255)

        return device_src.astype("float32")
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def regrain(src_img, out_col, deg):
        orig_h, orig_w, orig_c = src_img.get_raw().shape
        out_res = out_col.reshape(orig_h, orig_w, orig_c).astype(np.float32)
        # out_res = rgb_out.reshape(256, 256, 3).astype(np.float32)
        octave.addpath(octave.genpath('.'))
        octave.eval("warning('off','Octave:shadowed-function')")
        octave.eval('pkg load image')
        octave.eval('pkg load statistics')
        out_raw = octave.regrain(src_img.get_raw() * 255, out_res * 255, deg) / 255
        
        #out_res = out_raw.reshape(256 * 256, 3).astype(np.float32)
        out_res = out_raw.reshape(orig_h * orig_w, orig_c).astype(np.float32)

        return out_res