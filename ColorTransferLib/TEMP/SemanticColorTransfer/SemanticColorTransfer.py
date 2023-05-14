"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import json
import h5py
import cv2
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: ...
#   Author: Herbert Potechius
#   Published in: ...
#   Year of Publication: 2023
#
# Abstract:
#   ...
#
# Link: ...
# Source: ...
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class SemanticColorTransfer:
    identifier = "SemanticColorTransfer"
    title = "..."
    year = 2023

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # HOST METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "SemanticColorTransfer",
            "title": "...",
            "year": 2023,
            "abstract": "TODO"
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src_rgb, src_sem, src_depth, ref_rgb, ref_sem, ref_depth, opt):
        print("qHello")
        # 1. Separate image into depth based layers 
    # ------------------------------------------------------------------------------------------------------------------
    # define tone mapping for HDR images
    # ------------------------------------------------------------------------------------------------------------------
    def tonemapping(rgb_color, mask):
        render_entity_id = mask.astype("int32")[:,:,0]
        gamma                             = 1.0/2.2   # standard gamma correction exponent
        inv_gamma                         = 1.0/gamma
        percentile                        = 90        # we want this percentile brightness value in the unmodified image...
        brightness_nth_percentile_desired = 0.8       # ...to be this bright after scaling
        valid_mask = render_entity_id != -1

        if np.count_nonzero(valid_mask) == 0:
            scale = 1.0 # if there are no valid pixels, then set scale to 1.0
        else:
            brightness       = 0.3*rgb_color[:,:,0] + 0.59*rgb_color[:,:,1] + 0.11*rgb_color[:,:,2] # "CCIR601 YIQ" method for computing brightness
            brightness_valid = brightness[valid_mask]

            eps                               = 0.0001 # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
            brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

            if brightness_nth_percentile_current < eps:
                scale = 0.0
            else:
                scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

        rgb_color_tm = np.power(np.maximum(scale*rgb_color,0), gamma)
        rgb_color_tm = np.clip(rgb_color_tm, 0.0, 1.0)

        return rgb_color_tm

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # define semantic mapping
    mapping = np.array([ [0                , 0                , 0],
            [174              , 199              , 232],
            [152              , 223              , 138],
            [31               , 119              , 180],
            [255              , 187              , 120],
            [188              , 189              , 34],
            [140              , 86               , 75],
            [255              , 152              , 150],
            [214              , 39               , 40],
            [197              , 176              , 213],
            [148              , 103              , 189],
            [196              , 156              , 148],
            [23               , 190              , 207],
            [178              , 76               , 76],
            [247              , 182              , 210],
            [66               , 188              , 102],
            [219              , 219              , 141],
            [140              , 57               , 197],
            [202              , 185              , 52],
            [51               , 176              , 203],
            [200              , 54               , 131],
            [92               , 193              , 61],
            [78               , 71               , 183],
            [172              , 114              , 82],
            [255              , 127              , 14],
            [91               , 163              , 138],
            [153              , 98               , 156],
            [140              , 153              , 101],
            [158              , 218              , 229],
            [100              , 125              , 154],
            [178              , 127              , 135],
            [120              , 185              , 128],
            [146              , 111              , 194],
            [44               , 160              , 44],
            [112              , 128              , 144],
            [96               , 207              , 209],
            [227              , 119              , 194],
            [213              , 92               , 176],
            [94               , 106              , 211],
            [82               , 84               , 163],
            [100              , 85               , 144]], dtype=np.int64 )

    # Read options file
    with open("ColorTransferLib/Options/SemanticColorTransfer.json", 'r') as f:
        options = json.load(f)
    
    layers = []

    # Read source and reference RGB images
    img = h5py.File("data/ai_048_001_cam_00.hdf5", 'r')['dataset']
    src_rgb = np.float32(SemanticColorTransfer.tonemapping(img[50,:,:,:3], img[50,:,:,28:29]))
    ref_rgb = np.float32(SemanticColorTransfer.tonemapping(img[99,:,:,:3], img[99,:,:,28:29]))
    print(img.shape)

    cv2.imwrite("/home/potechius/Downloads/SemanticColorTransfer/source.png", cv2.cvtColor(src_rgb, cv2.COLOR_BGR2RGB ) * 255)
    cv2.imwrite("/home/potechius/Downloads/SemanticColorTransfer/reference.png", cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2RGB ) * 255)
    exit()

    # Read reflectance images
    src_ref = np.float32(SemanticColorTransfer.tonemapping(img[50,:,:,6:9], img[50,:,:,28:29]))
    src_sha = np.float32(SemanticColorTransfer.tonemapping(img[50,:,:,3:6], img[50,:,:,28:29]))
    #cv2.imwrite("/home/potechius/Downloads/out_temp/src_sha.png", cv2.cvtColor(src_sha, cv2.COLOR_BGR2RGB ) * 255)
    #exit()

    # Read depth images
    src_depth = np.float32(img[50,:,:,12:13])
    # src_depth = np.clip(src_depth * 25.5, 0, 255)
    # print(src_depth.shape)
    # print(np.max(src_depth))
    # cv2.imwrite("/home/potechius/Downloads/out_temp/src_depth.png", src_depth)
    # exit()

    # Read semantic images
    src_sem = np.float32(img[50,:,:,29:30])
    src_sem[src_sem == -1] = 0
    #src_sem = mapping[src_sem]
    #src_sem = np.float32(src_sem[:,:,0,:])
    
    # Read instace images
    src_ins = np.float32(img[50,:,:,30:31])
    src_ins[src_ins == -1] = 0

    # combine semantic and instance
    # the resulting id is calculated by: semantic id * 1000 + instance id
    # Example:
    # instance id = 101; Semantic id = 33; final id = 10133
    src_semins = np.int64(np.add(src_ins*100, src_sem))
    #print(src_semins[100, 100, 0])

    # Get a list of unique elements
    src_unique = np.unique(src_semins)
    #print(src_unique)

    # Get (id=22) Ceiling, (id=2) Floor and (id=1) Walls
    # - create mask
    # - cut mask from reflectance
    mask_ceiling = np.where(src_semins == 22, 1, 0)
    mask_floor = np.where(src_semins == 2, 1, 0)
    mask_wall = np.where(src_semins == 1, 1, 0)
    mask_room = np.add(np.add(mask_ceiling, mask_floor), mask_wall)
    mask_room = cv2.erode(np.float32(mask_room), np.ones((5, 5), np.uint8)) 
    mask_room = np.expand_dims(mask_room, axis=2)
    print(mask_room.shape)
    mask_ref = np.multiply(src_ref, mask_room)

    # Inpainting
    inpaint_mask = np.uint8((1-mask_room) * 255)
    inpaint_ref = np.uint8(mask_ref * 255)
    print(np.max(inpaint_ref))
    
    print(mask_ref.dtype)
    print(inpaint_mask.dtype)
    print(mask_ref.shape)
    print(inpaint_mask.shape)
    dst = cv2.inpaint(inpaint_ref, inpaint_mask, 3,cv2.INPAINT_TELEA)

    cv2.imwrite("/home/potechius/Downloads/out_temp/masks/inpaint_mask.png", inpaint_mask)
    cv2.imwrite("/home/potechius/Downloads/out_temp/masks/inpaint.png", dst)



    exit()

    for elem in src_unique:
        mask = np.where(src_semins == elem, 1, 0)

        mask_ref = np.multiply(src_ref, mask)
        cv2.imwrite("/home/potechius/Downloads/out_temp/masks/"+str(elem)+".png", mask_ref * 255)
    exit()



    # Iterate over each instance and seperate them

    # cv2.imwrite("/home/potechius/Downloads/out_temp/src_semin.png", src_semin)

    # combine 



    max_inst = np.max(src_semin)
    for i in range(int(max_inst)):
        iin_layer = {
            "depth": None
        }
        print(i)
        print(np.count_nonzero(src_semin == i))
        print("-----------")
        # mask = src_semin
        # mask[mask != i] = 0
        # mask[mask == i] = 1
        # cv2.imshow('image',np.float32(mask) )
        # cv2.waitKey(0)
    exit()


    

    # Show images
    #cv2.imshow('image',np.float32(src_sem[:,:,0,:] / 255))
    cv2.imshow('image',src_semin )
    cv2.waitKey(0)

    #sm = SemanticColorTransfer()
    #print(options)