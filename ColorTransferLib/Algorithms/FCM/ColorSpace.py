import numpy as np
import cv2
import copy

class ColorSpace():

    color_terms = np.array(["Red", "Yellow", "Green", "Blue", "Black", "White", "Grey", "Orange", "Brown", "Pink", "Purple"])
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def CAT_polar_to_cartesian(hsv_colors):
        hsv_colors_copy = copy.deepcopy(hsv_colors)
        for c in ColorSpace.color_terms:
            if hsv_colors_copy[c].shape[0] == 0:
                continue
            hsv_colors_copy[c] = ColorSpace.polar_to_cartesian(hsv_colors_copy[c])
        return hsv_colors_copy
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def polar_to_cartesian(hsv_colors):
        hue_angle = hsv_colors[:,0]
        sat_radius = hsv_colors[:,1]
        value = hsv_colors[:,2]

        hue_angle = (360 + hue_angle) % 360
        #sat_radius = np.clip(sat_radius, 0, 255)
        #value = np.clip(value, 0, 255)


        # weighting if the radius in order to get the HSV-cone
        weighted_radius = sat_radius * (value / 255.0)

        x_pos = weighted_radius * np.cos(np.radians(hue_angle))
        y_pos = weighted_radius * np.sin(np.radians(hue_angle))
        z_pos = value

        hsv_cart = np.concatenate((np.expand_dims(x_pos,1), np.expand_dims(y_pos,1), np.expand_dims(z_pos,1)), axis=1)
        return hsv_cart

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def CAT_cartesian_to_polar(hsv_colors):
        hsv_colors_copy = copy.deepcopy(hsv_colors)
        for c in ColorSpace.color_terms:
            if hsv_colors_copy[c].shape[0] == 0:
                continue
            # print(np.min(hsv_colors_copy[c][:,0]))
            # print(np.max(hsv_colors_copy[c][:,0]))
            # print(np.min(hsv_colors_copy[c][:,1]))
            # print(np.max(hsv_colors_copy[c][:,1]))
            # print(np.min(hsv_colors_copy[c][:,2]))
            # print(np.max(hsv_colors_copy[c][:,2]))
            # print("\n")
            hsv_colors_copy[c] = ColorSpace.cartesian_to_polar(hsv_colors_copy[c])
        return hsv_colors_copy

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def cartesian_to_polar(hsv_colors):
        x = hsv_colors[:,0]
        y = hsv_colors[:,1]
        z = hsv_colors[:,2]

        # if z == 0:
        #     hue = 0
        #     sat = 0
        #     val = 0
        # else:
        # weighting if the radius in order to get the HSV-cone
        radius_weighting = 255.0 / (z + 0.000001) 

        #x *= radius_weighting
        #y *= radius_weighting


        # hue has to be converted to degrees
        # Note: hue is in range [-180, 180] -> value smaller than 0 hat to be mapped to [180, 360]
        hue = np.degrees(np.arctan2(y, x))

        # NOTE: Maybe important
        # for i, val in enumerate(zip(np.arctan2(y, x),y)):
        #     if val[0] >= 0 and val[1] < 0:
        #         hue[i] +=360
        #     elif val[0] < 0:
        #         hue[i] +=360

        hue = (hue + 360) % 360

        sat = np.sqrt(x ** 2 + y ** 2) * radius_weighting
        val = z

        hsv_polar = np.concatenate((np.expand_dims(hue,1), np.expand_dims(sat,1), np.expand_dims(val,1)), axis=1)
        return hsv_polar
    
    # ------------------------------------------------------------------------------------------------------------------
    # Transfers RGB values to cartesian HSV
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def RGB2cartHSV(rgb):
        # if src_color is in range [0,1] the SV channels are also in range [0,1] but H channel is in range [0,360]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)[:,0,:]
        hsv[:,1:3] = hsv[:,1:3] * 255
        hsv_cart = ColorSpace.polar_to_cartesian(hsv)
        return hsv_cart
    
    # ------------------------------------------------------------------------------------------------------------------
    # Transfers RGB values to cylindric HSV
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def RGB2HSV(rgb):
        # if src_color is in range [0,1] the SV channels are also in range [0,1] but H channel is in range [0,360]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)[:,0,:]
        hsv[:,1:3] = hsv[:,1:3] * 255
        hsv_cart = hsv
        return hsv_cart
    
    # ------------------------------------------------------------------------------------------------------------------
    # Transfers RGB values to cartesian HSV
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def HSV2cartRGB(hsv):
        #hsv[:,1:3] = hsv[:,1:3] * 255
        hsv_polar = ColorSpace.cartesian_to_polar(hsv)
        # if src_color is in range [0,1] the SV channels are also in range [0,1] but H channel is in range [0,360]
        hsv_polar[:,1:3] = hsv_polar[:,1:3] / 255
        # returns normalized rgb values in range [0, 1]
        rgb = cv2.cvtColor(np.expand_dims(hsv_polar, axis=1), cv2.COLOR_HSV2RGB_FULL)

        return rgb
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def CAT_HSV2cartRGB(hsv_colors):
        hsv_colors_copy = copy.deepcopy(hsv_colors)
        for c in ColorSpace.color_terms:
            if hsv_colors_copy[c].shape[0] == 0:
                continue
            hsv_colors_copy[c] = ColorSpace.HSV2cartRGB(hsv_colors_copy[c]).squeeze() * 255
        return hsv_colors_copy