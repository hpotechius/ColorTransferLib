"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from numba import cuda
import math
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.BaseOptions import BaseOptions
import cv2
import sys
from ColorTransferLib.ImageProcessing.Image import Image
import json


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: ...
#   Author: ...
#   Published in: ...
#   Year of Publication: ...
#
# Abstract:
#   ...
#
# Link: ...
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class FrequencyColorTransfer:
    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }
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
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "FrequencyColorTransfer",
            "title": "...",
            "year": 2022,
            "abstract": "..."
        }

        return info

    @staticmethod
    def getFrequencyImage(img, seperateChannels=False):
        if seperateChannels:
            mag = np.zeros_like(img)
            phase = np.zeros_like(img)
            for i in range(3):
                img_dft = np.fft.fft2(np.float32(img[:,:,i]))
                dft_shift = np.fft.fftshift(img_dft)
                mag_temp, phase_temp = cv2.cartToPolar(dft_shift.real, dft_shift.imag)
                mag[:,:,i] = mag_temp
                phase[:,:,i] = phase_temp
        else:
            img_dft = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            # convert image to floats and do dft saving as complex output
            img_dft = np.fft.fft2(np.float32(img_dft))

            # apply shift of origin from upper left corner to center of image
            dft_shift = np.fft.fftshift(img_dft)

            # extract magnitude and phase images
            mag, phase = cv2.cartToPolar(dft_shift.real, dft_shift.imag)

        return mag, phase
    
    @staticmethod
    def showSpectrum(name, mag):
        mag = np.where(mag <= 0.0, 0.0000000000001, mag) 
        mag = np.log(mag) / 30.0 
        maxV = np.max(mag)
        minV = np.min(mag)
        #print(minV)
        #print(maxV)
        #cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
        #mag = mag.astype('uint8')
        return mag
        

    @staticmethod
    def getSpatialImage(mag, phase,seperateChannels=False):
        #maxV = np.max(mag)
        #minV = np.min(mag)

        #mag = mag.astype('float32')
        #cv2.normalize(mag, mag, minV, maxV, cv2.NORM_MINMAX)
        #mag = np.exp(mag * 30)
       

        # NEW CODE HERE: raise mag to some power near 1
        # values larger than 1 increase contrast; values smaller than 1 decrease contrast
        #mag = cv.pow(mag, 0.9)


        # convert magnitude and phase into cartesian real and imaginary components
        real, imag = cv2.polarToCart(mag, phase)

        # combine cartesian components into one complex image
        back = cv2.merge([real, imag])

        asdf = np.zeros_like(mag, dtype="complex128")
        asdf.real = real
        asdf.imag = imag

        # shift origin from center to upper left corner
        #back_ishift = np.fft.ifftshift(back)

        # do idft saving as complex output
        #img_back = cv2.idft(back_ishift)

        # combine complex components into original image again
        #img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

        img_back = abs(np.fft.ifft2(asdf))

        # re-normalize to 8-bits
        #min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
        #print(min,max)


        #print(np.min(img_back))
        #print(np.max(img_back))
        #exit()

        #img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #img_back = np.clip(img_back, 0, 255)
        #img_back = (img_back - np.min(img_back)) / np.max(img_back) * 255

        return img_back

    @staticmethod
    def getArea(radius, width, mag):
        hh, ww = mag.shape[:2]
        #print(hh)
        #print(ww)
        hh2 = hh // 2
        ww2 = ww // 2
        # define circles
        yc = hh2
        xc = ww2
        # draw filled circle in white on black background as mask
        #print(np.min(mag))
        #print(np.max(mag))
        
        mag = np.log(mag) / 30.0 * 255
        #print(np.max(mag))
        #print(mag.shape)
        mag = mag.astype('uint8')

        mask = np.ones_like(mag) * 255
        mask = cv2.circle(mask, (xc,yc), radius, (0,0,0), -1)
        # apply mask to image
        if radius > 0:
            mag = cv2.bitwise_and(mag, mask)

        mask2 = np.zeros_like(mag)
        mask2 = cv2.circle(mask2, (xc,yc), radius + width, (255,255,255), -1)
        # apply mask to image
        mag = cv2.bitwise_and(mag, mask2)

        mag = mag.astype('float32')
        # substract minus one in order to set the minimum to zero
        mag = np.exp(mag / 255.0 * 30) - 1.0
        return mag

    @staticmethod
    def setArea(mag, radius, width, mag_over):
        hh, ww = mag.shape[:2]
        yc = hh // 2
        xc = ww // 2
        # draw filled circles in white on black background as masks
        mask1 = np.zeros_like(mag)
        mask1 = cv2.circle(mask1, (xc,yc), radius, (255,255,255), -1)
        mask2 = np.zeros_like(mag)
        mask2 = cv2.circle(mask2, (xc,yc), radius + width, (255,255,255), -1)

        # subtract masks and make into single channel
        if radius > 0:
            mask = cv2.subtract(mask2, mask1) / 255
        else:
            mask = mask2 / 255

        dst = mag_over * mask + mag * (np.ones_like(mag) - mask)

        #dst = cv2.addWeighted(mag_over, 0.1, mag, 0,9)

        #FrequencyColorTransfer.showSpectrum("t2", dst[:,:,0])
        #FrequencyColorTransfer.showSpectrum("t3", mag_over[:,:,0])
        

        return dst



    @staticmethod
    def globalColorTransfer(src, ref):
        mean_src = np.mean(src, axis=(0,1))
        mean_ref = np.mean(ref, axis=(0,1))
        std_src = np.std(src, axis=(0,1))
        std_ref = np.std(ref, axis=(0,1))

        out = (src - mean_src) * (std_ref / std_src) + mean_ref
        return np.clip(out, 0, 255)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, options=[]):
        opt = BaseOptions(options)

        # generate Gaussina pyramid
        img1_copy = src.copy()
        gp_img1 = [img1_copy]

        for i in range(6):
            #print(img1_copy.shape)
            img1_copy = cv2.pyrDown(img1_copy)
            gp_img1.append(img1_copy)

        #gaus_pyr.append(gp_img1)
        #print("---")

        # generate Laplacian Pyramid
        img1_copy = gp_img1[5]
        lp_img1 = [img1_copy]
        for i in range(5, 0, -1):
            gaussian_expanded = cv2.pyrUp(gp_img1[i])
            #print(gaussian_expanded.shape)
            #print(gp_img1[i-1].shape)
            laplacian = cv2.subtract(gp_img1[i-1], cv2.resize(gaussian_expanded, (gp_img1[i-1].shape[1], gp_img1[i-1].shape[0])))
            lp_img1.append(laplacian)

        ##########################################
        # generate Gaussina pyramid
        img2_copy = ref.copy()
        gp_img2 = [img2_copy]

        for i in range(6):
            #print(img1_copy.shape)
            img2_copy = cv2.pyrDown(img2_copy)
            gp_img2.append(img2_copy)

        # generate Laplacian Pyramid
        img2_copy = gp_img2[5]
        lp_img2 = [img2_copy]
        for i in range(5, 0, -1):
            gaussian_expanded = cv2.pyrUp(gp_img2[i])
            #print(gaussian_expanded.shape)
            #print(gp_img1[i-1].shape)
            laplacian = cv2.subtract(gp_img2[i-1], cv2.resize(gaussian_expanded, (gp_img2[i-1].shape[1], gp_img2[i-1].shape[0])))
            lp_img2.append(laplacian)


        # COLOR TRANSFER
        o0 = lp_img1[0]#FrequencyColorTransfer.globalColorTransfer(lp_img1[0], lp_img2[0])
        o1 = FrequencyColorTransfer.globalColorTransfer(lp_img1[1], lp_img2[1])
        o2 = FrequencyColorTransfer.globalColorTransfer(lp_img1[2], lp_img2[2])
        o3 = FrequencyColorTransfer.globalColorTransfer(lp_img1[3], lp_img2[3])
        o4 = FrequencyColorTransfer.globalColorTransfer(lp_img1[4], lp_img2[4])
        o5 = FrequencyColorTransfer.globalColorTransfer(lp_img1[5], lp_img2[5])


        #src = cv2.resize(src, (src.shape[1]//2, src.shape[0]//2 ), interpolation = cv2.INTER_AREA)
        #out = FrequencyColorTransfer.globalColorTransfer(src, ref)
        h = 256 
        w = 256

        out_top = np.hstack((cv2.resize(lp_img1[0], (w, h)), cv2.resize(lp_img1[1], (w, h))))
        out_top = np.hstack((out_top, cv2.resize(lp_img1[2], (w, h))))
        out_top = np.hstack((out_top, cv2.resize(lp_img1[3], (w, h))))
        out_top = np.hstack((out_top, cv2.resize(lp_img1[4], (w, h))))
        out_top = np.hstack((out_top, cv2.resize(lp_img1[5], (w, h))))

        out_btm = np.hstack((cv2.resize(lp_img2[0], (w, h)), cv2.resize(lp_img2[1], (w, h))))
        out_btm = np.hstack((out_btm, cv2.resize(lp_img2[2], (w, h))))
        out_btm = np.hstack((out_btm, cv2.resize(lp_img2[3], (w, h))))
        out_btm = np.hstack((out_btm, cv2.resize(lp_img2[4], (w, h))))
        out_btm = np.hstack((out_btm, cv2.resize(lp_img2[5], (w, h))))

        out_fin = np.hstack((cv2.resize(o0, (w, h)), cv2.resize(o1, (w, h))))
        out_fin = np.hstack((out_fin, cv2.resize(o2, (w, h))))
        out_fin = np.hstack((out_fin, cv2.resize(o3, (w, h))))
        out_fin = np.hstack((out_fin, cv2.resize(o4, (w, h))))
        out_fin = np.hstack((out_fin, cv2.resize(o5, (w, h))))

        out = np.vstack((out_top, out_btm))
        out = np.vstack((out, out_fin))

        finalo = cv2.resize(o0, (w, h)) + cv2.resize(o1, (w, h)) + cv2.resize(o2, (w, h)) + cv2.resize(o3, (w, h)) + cv2.resize(o4, (w, h)) + cv2.resize(o5, (w, h))

        cv2.imshow("Back", cv2.cvtColor(out,cv2.COLOR_RGB2BGR) / 255.0)
        cv2.imshow("FIN", cv2.cvtColor(finalo,cv2.COLOR_RGB2BGR) / 255.0)
        cv2.waitKey(0)



        exit()



        # [1] Get Frequency image of src and ref
        mag_src, phase_src = FrequencyColorTransfer.getFrequencyImage(src, seperateChannels=True)
        mag_ref, phase_ref = FrequencyColorTransfer.getFrequencyImage(ref, seperateChannels=True)

        """
        XXX = np.zeros_like(src)
        XXX[:,:,0] = FrequencyColorTransfer.getSpatialImage(mag_src[:,:,0], phase_src[:,:,0])
        XXX[:,:,1] = FrequencyColorTransfer.getSpatialImage(mag_src[:,:,1], phase_src[:,:,1])
        XXX[:,:,2] = FrequencyColorTransfer.getSpatialImage(mag_src[:,:,2], phase_src[:,:,2])

        cv2.imshow("source", cv2.cvtColor(src,cv2.COLOR_RGB2BGR)  /255)
        cv2.imshow("out", cv2.cvtColor(XXX,cv2.COLOR_RGB2BGR)  /255)
        cv2.waitKey(0)
        exit()
        """
        #print(mag[640,383])
        #FrequencyColorTransfer.showSpectrum("t1", mag)

        # [2] Iterate over bands and apply color transfer for each channel
        out_mag = np.zeros_like(src)
        out_phase = np.zeros_like(src)
        out_img = np.zeros_like(src)
        width = 1
        diagonal = math.ceil(math.sqrt(pow(src.shape[0] // 2, 2) + pow(src.shape[1] // 2, 2)))
        for r in range(0, diagonal, width):
            print(r)
            band_img_src = np.zeros_like(src)
            band_img_ref = np.zeros_like(ref)
            for c in range(3):
                mag_src_c = mag_src[:,:,c]                
                mag_src_c = FrequencyColorTransfer.getArea(r, width, mag_src_c)
                img_src_c = FrequencyColorTransfer.getSpatialImage(mag_src_c, phase_src[:,:,c])
                band_img_src[:,:,c] = img_src_c
                #cv2.imshow("d",img_src_c)

                mag_ref_c = mag_ref[:,:,c]
                mag_ref_c = FrequencyColorTransfer.getArea(r, width, mag_ref_c)
                img_ref_c = FrequencyColorTransfer.getSpatialImage(mag_ref_c, phase_ref[:,:,c])
                band_img_ref[:,:,c] = img_ref_c
                #FrequencyColorTransfer.showSpectrum("t2", mag_src_c)
                #cv2.imshow("d",img_ref_c)
                #cv2.waitKey(0)

            out = FrequencyColorTransfer.globalColorTransfer(band_img_src, band_img_ref)
            #out = band_img_src 

            mag_out, phase_out = FrequencyColorTransfer.getFrequencyImage(out, seperateChannels=True)
            #mag_out, phase_out = FrequencyColorTransfer.getFrequencyImage(band_img_src, seperateChannels=True)
            #FrequencyColorTransfer.showSpectrum("t2", mag_out[:,:,0])

            #dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(out[:,:,0]))
            #back = abs(np.fft.ifft2(dark_image_grey_fourier))
            #print(dark_image_grey_fourier.dtype)
            #print(dark_image_grey_fourier.shape)
            #print(dark_image_grey_fourier[0,0])
            #cv2.imshow("daw", back / 255)
            #cv2.waitKey(0)
            #exit()


            #testo = np.zeros_like(src)
            #testo[:,:,0] = FrequencyColorTransfer.getSpatialImage(mag_out[:,:,0], phase_out[:,:,0])
            #testo[:,:,1] = FrequencyColorTransfer.getSpatialImage(mag_out[:,:,1], phase_out[:,:,1])
            #testo[:,:,2] = FrequencyColorTransfer.getSpatialImage(mag_out[:,:,2], phase_out[:,:,2])
            #cv2.imshow("ou", cv2.cvtColor(out,cv2.COLOR_RGB2BGR) / 255.0)
            #cv2.imshow("oui", cv2.cvtColor(testo,cv2.COLOR_RGB2BGR) / 255.0)            
            #cv2.imshow("ou", out[:,:,0] / 255.0)
            #cv2.imshow("oui", testo[:,:,0] / 255.0)            
            #cv2.waitKey(0)
            #exit()

            #print(phase_out.shape)
            #print(np.min(phase_out))
            #print(np.max(phase_out))
            #cv2.imshow("dawd", phase_out / 6)
            #cv2.waitKey(0)
            #exit()

            #FrequencyColorTransfer.showSpectrum("t1", mag_out[:,:,0])
            #mag_out[:,:,0] = FrequencyColorTransfer.getArea(r, width, mag_out[:,:,0])
            #mag_out[:,:,1] = FrequencyColorTransfer.getArea(r, width, mag_out[:,:,1])
            #mag_out[:,:,2] = FrequencyColorTransfer.getArea(r, width, mag_out[:,:,2])
            
            #FrequencyColorTransfer.showSpectrum("t2", mag_out[:,:,0])

            out_mag = FrequencyColorTransfer.setArea(out_mag, r, width, mag_out)

            #FrequencyColorTransfer.showSpectrum("t2", mask[:,:,0])
            #cv2.imshow("kk", mask * 25)
           #cv2.waitKey(0)
            #FrequencyColorTransfer.showSpectrum("XXX", out_mag[:,:,0])

            XXX = np.zeros_like(src)
            XXX[:,:,0] = FrequencyColorTransfer.getSpatialImage(out_mag[:,:,0], phase_src[:,:,0])
            XXX[:,:,1] = FrequencyColorTransfer.getSpatialImage(out_mag[:,:,1], phase_src[:,:,1])
            XXX[:,:,2] = FrequencyColorTransfer.getSpatialImage(out_mag[:,:,2], phase_src[:,:,2])
            #XXX[:,:,0] = FrequencyColorTransfer.getSpatialImage(mag_out[:,:,0], phase_out[:,:,0])
            #XXX[:,:,1] = FrequencyColorTransfer.getSpatialImage(mag_out[:,:,1], phase_out[:,:,1])
            #XXX[:,:,2] = FrequencyColorTransfer.getSpatialImage(mag_out[:,:,2], phase_out[:,:,2])

            source00 = cv2.cvtColor(cv2.resize(band_img_src, (400,250)),cv2.COLOR_RGB2BGR) / 255.0
            ref00 = cv2.cvtColor(cv2.resize(band_img_ref, (400,250)),cv2.COLOR_RGB2BGR) / 255.0
            sopec00 = cv2.cvtColor(cv2.resize(FrequencyColorTransfer.showSpectrum("t3", out_mag[:,:,0]), (400,250)), cv2.COLOR_GRAY2BGR)
            out00 = cv2.cvtColor(cv2.resize(XXX, (400,250)),cv2.COLOR_RGB2BGR) / 255.0

            vis1 = np.concatenate((source00, ref00), axis=0)
            vis2 = np.concatenate((sopec00, out00), axis=0)
            vis = np.concatenate((vis1, vis2), axis=1)
            cv2.imshow("out", vis)
            #cv2.imshow("Ref", cv2.cvtColor(band_img_ref,cv2.COLOR_RGB2BGR) / 255.0)
            #cv2.imshow("Output", cv2.cvtColor(XXX,cv2.COLOR_RGB2BGR) / 255.0)
            #cv2.imshow("OutputOrig", cv2.cvtColor(out,cv2.COLOR_RGB2BGR) / 255.0)
            cv2.waitKey(0)
            exit()


        #print(mag[640,383])

        #FrequencyColorTransfer.showSpectrum("t2", mag)
        
        #img_back = FrequencyColorTransfer.getSpatialImage(mag, phase)
    

        #cv2.imshow("Original", cv2.cvtColor(src,cv2.COLOR_RGB2BGR) / 255.0)
        #cv2.imshow("Back", img_back / 255.0)

        #print("FF")
        exit()


        return out

if __name__  == "__main__":
    src_img = "data/images/WhiteRose.jpg"
    ref_img = "data/images/starry-night.jpg"

    with open("module/Options/FrequencyColorTransfer.json", 'r') as f:
        options = json.load(f)

    src = Image(file_path=src_img)
    ref = Image(file_path=ref_img)

    src_color = src.get_raw() * 255.0
    ref_color = ref.get_raw() * 255.0
    output = FrequencyColorTransfer.apply(src_color, ref_color, options)
