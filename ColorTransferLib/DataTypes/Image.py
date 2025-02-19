"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Reads a color (RGB) or a greyscale (Grey) image and converts it to a 32 bit RGB image with a value range of [0, 1]
# Possible image color formats: ["RGB", "Grey", "CIELab", "lab"]. CIELab and lab are only available after conversion.
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Image:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # If <file_path> is None, an empty image with the given <size>=(width, height) will be created
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, file_path=None, array=None, size=(0, 0), color="RGB", normalized=False):
        if file_path is None and array is None:
            self.__img = np.zeros((size[1], size[0]), dtype=np.float32)
        elif file_path is not None and array is None:
            self.__img = cv2.imread(file_path)
            #self.__img = cv2.resize(self.__img, (self.__img.shape[1] //3, self.__img.shape[0] //3))
        elif file_path is None and array is not None:
            self.__img = array.astype(np.float32)
        else:
            raise ValueError("file_path or array has to be None")

        self.__type = "Image"
        self.__color_format = color
        self.__width = self.__img.shape[1]
        self.__height = self.__img.shape[0]
        self.__pixelnum = self.__img.shape[0] * self.__img.shape[1]

        if color == "RGB":
            self.__img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2RGB).astype(np.float32)
        elif color == "BGR":
            self.__img = self.__img.astype(np.float32)
        elif color == "Grey":
            self.__img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2RGB).astype(np.float32)
        else:
            raise ValueError(color + " is not a valid color format.")

        if not normalized:
            self.__img = self.__img / 255.0

        # WARNING: This is slow. Temporarily removed
        #self.__3D_color_histogram = self.__calculate_3D_color_histogram()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Writes the file to the specified path.
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, out_path):
        out_img = (self.__img * 255.0).astype(np.uint8)
        cv2.imwrite(out_path + ".png", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------------------------------------------------------
    # Converts the image to the given color space.
    # ------------------------------------------------------------------------------------------------------------------
    def convert_to(self, color_format):
        if self.__color_format == color_format:
            print("INFO: Image has already the color format " + color_format + ".")

    # ------------------------------------------------------------------------------------------------------------------
    # Resizes the image without keeping the aspect ratio.
    # ------------------------------------------------------------------------------------------------------------------
    def resize(self, width=100, height=100):
        self.__width = width
        self.__height = height
        self.__img = cv2.resize(self.__img, (width, height), interpolation=cv2.INTER_AREA)
        #print(self.__width)
        #print(self.__width)

    # ------------------------------------------------------------------------------------------------------------------
    # Shows the image using OpenCV. The showed image can be scaled by providing a <resize>-values. If the
    # <stretch>-value is False, the image keeps its aspect ratio and will be extended by black borders fulfill the
    # given <resize>-values. If the resized image has to be post-processed without visualization, the <show>-value
    # has to be set to False.
    # ------------------------------------------------------------------------------------------------------------------
    def show(self, resize=(500, 500), stretch=False, show=True):
        if stretch:
            img_resized = cv2.resize(self.__img, (resize[0], resize[1]), interpolation=cv2.INTER_AREA)
        if not stretch:
            in_ratio = self.__width / self.__height
            out_ratio = resize[0] / resize[1]
            scale_factor = in_ratio / out_ratio
            if scale_factor < 1:
                top = bottom = 0
                left = right = int((self.__width / scale_factor - self.__width) / 2)
            elif scale_factor > 1:
                left = right = 0
                top = bottom = int((self.__height * scale_factor - self.__height) / 2)
            else:
                top = bottom = left = right = 0
            img_resized = cv2.copyMakeBorder(self.__img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0,0,0))
            img_resized = cv2.resize(img_resized, (resize[0], resize[1]), interpolation=cv2.INTER_AREA)

        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        if show:
            cv2.imshow('image', img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            return img_resized

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_type(self):
        return self.__type

    # ------------------------------------------------------------------------------------------------------------------
    # return the color values as vector
    # ------------------------------------------------------------------------------------------------------------------
    def get_colors(self):
        return self.__img.reshape(self.__width * self.__height, 1, 3)

    # ------------------------------------------------------------------------------------------------------------------
    # returns the image as numpy array
    # ------------------------------------------------------------------------------------------------------------------
    def get_raw(self):
        return self.__img

    # ------------------------------------------------------------------------------------------------------------------
    # returns the color histogram, mean and variance
    # ------------------------------------------------------------------------------------------------------------------
    def get_color_statistic(self, bins=256, normalized=False):
        color = self.get_colors()
        rgb_c = (color * 255.0).astype(int).reshape(color.shape[0], color.shape[2])
        histo_red = np.asarray(np.histogram(rgb_c[:,0], bins=np.arange(bins+1))[0]).reshape(bins,1)
        histo_green = np.asarray(np.histogram(rgb_c[:,1], bins=np.arange(bins+1))[0]).reshape(bins,1)
        histo_blue = np.asarray(np.histogram(rgb_c[:,2], bins=np.arange(bins+1))[0]).reshape(bins,1)

        if normalized:
            histo_red = histo_red / np.sum(histo_red)
            histo_green = histo_green / np.sum(histo_green)
            histo_blue = histo_blue / np.sum(histo_blue)

        histo = np.concatenate((histo_red, histo_green, histo_blue), axis=1)
        mean = np.mean(rgb_c, axis=0).astype(int)
        std = np.std(rgb_c, axis=0).astype(int)
        return histo, mean, std
    
    # ------------------------------------------------------------------------------------------------------------------
    # returns 
    # ------------------------------------------------------------------------------------------------------------------
    def get_color_distribution(self):
        color = self.get_colors()
        color = color[np.random.randint(color.shape[0], size=5000), :]
        rgb_c = (color * 255.0).astype(int).reshape(color.shape[0], color.shape[2])
        rgb_c = np.unique(rgb_c, axis=0)
        return rgb_c

    # ------------------------------------------------------------------------------------------------------------------
    # returns the 3D color histogram
    # ------------------------------------------------------------------------------------------------------------------
    def get_color_statistic_3D(self, bins=[256,256,256], normalized=False):
        color = self.get_colors()
        rgb_c = (color * 255.0).astype(int).reshape(color.shape[0], color.shape[2])
        histo = np.asarray(np.histogramdd(rgb_c, bins)[0])

        if normalized:
            sum_h = np.sum(histo)
            histo /= sum_h
        return histo

    # ------------------------------------------------------------------------------------------------------------------
    # returns the quantized 3D color histogram
    # ------------------------------------------------------------------------------------------------------------------
    def get_3D_color_histogram(self):
        return self.__3D_color_histogram

    # ------------------------------------------------------------------------------------------------------------------
    # return image width
    # ------------------------------------------------------------------------------------------------------------------
    def get_width(self):
        return self.__width

    # ------------------------------------------------------------------------------------------------------------------
    # return image height
    # ------------------------------------------------------------------------------------------------------------------
    def get_height(self):
        return self.__height

    # ------------------------------------------------------------------------------------------------------------------
    # return number of pixels
    # ------------------------------------------------------------------------------------------------------------------
    def get_pixelnum(self):
        return self.__pixelnum

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # SETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # <colors> has to be provided as a vector, i.e., a flatten array
    # ------------------------------------------------------------------------------------------------------------------
    def set_colors(self, colors):
        self.__img = colors.reshape(self.__height, self.__width, 3).astype(np.float32)

    # ------------------------------------------------------------------------------------------------------------------
    # replaces the numpy array image
    # Parameters:
    # normalized = if True the array has to be normalized to range [0, 1], if false the range has to be [0, 255]
    # ------------------------------------------------------------------------------------------------------------------
    def set_raw(self, array, normalized=False):
        self.__img = array
        self.__img = self.__img.astype(np.float32)
        if not normalized:
            self.__img = self.__img / 255.0
        self.__width = self.__img.shape[1]
        self.__height = self.__img.shape[0]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # RGB color space is quantized in a 10x10x10 area 
    # ------------------------------------------------------------------------------------------------------------------
    def __calculate_3D_color_histogram(self):
        upd = np.clip(np.floor(self.__img * 10).astype(int), 0, 9).reshape(self.__height * self.__width, 3)
        uni = np.unique(upd, axis=0, return_counts=True)
        con = np.concatenate((uni[0], uni[1].reshape((-1, 1))), axis=1)
        return con
