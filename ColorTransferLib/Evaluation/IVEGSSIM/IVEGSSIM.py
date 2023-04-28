"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from skimage.metrics import structural_similarity as ssim
from torchmetrics import StructuralSimilarityIndexMeasure
import cv2
import math
import numpy as np
from scipy import signal
import sys
sys.path.insert(0, '/home/potechius/Projects/VSCode/ColorTransferLib/')
from ColorTransferLib.ImageProcessing.Image import Image
import time
from multiprocessing import Process, Pool, Manager, Semaphore
#import pysaliency

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Structural similarity index measure (SSIM)
# Measuring the perceived quality of an image regarding an original (uncompressed and distortion-free) image.
#
# Source: Image quality assessment: from error visibility to structural similarity
#
# Range [-1, 1]
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class IVEGSSIM:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def iso2Dgauss(size=11, sig=1.5, normalized=True):
        kernel = np.zeros((size,size))
        shift = size // 2
        for x in range(size): 
            for y in range(size): 
                x_s = x - shift
                y_s = y - shift
                kernel[y, x] = (1.0 / (2 * math.pi * math.pow(sig, 2))) * np.exp( -(math.pow(x_s,2)+math.pow(y_s,2)) / (2*math.pow(sig,2)) )
        
        if normalized:
            kernel /= np.sum(kernel)
        return kernel

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def ssim(src, ref):
        kernel_gaus = IVEGSSIM.iso2Dgauss()

        src_img = src
        ref_img = ref

        k1 = 0.01
        k2 = 0.03
        L = 1.0
        N = 11
        M = src_img.shape[0] * src_img.shape[1]
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        c3 = c2 / 2.0
        alp = 1.0
        bet = 1.0
        gam = 1.0
        w_r = 1.0/3.0
        w_g = 1.0/3.0
        w_b = 1.0/3.0
        
        mu_src = cv2.filter2D(src_img, -1, kernel_gaus)
        mu_ref = cv2.filter2D(ref_img, -1, kernel_gaus)


        src_pad = np.pad(src_img, ((5, 5), (5, 5), (0, 0)), 'reflect')
        ref_pad = np.pad(ref_img, ((5, 5), (5, 5), (0, 0)), 'reflect')

        sig_src = np.zeros_like(src_img)
        sig_ref = np.zeros_like(ref_img)
        cov = np.zeros_like(src_img)

        # Version 3
        # src_pad_ext.shape = (512, 512, 1, 11, 11, 3)
        src_pad_ext = np.lib.stride_tricks.sliding_window_view(src_pad, (11,11,3)).squeeze()
        ref_pad_ext = np.lib.stride_tricks.sliding_window_view(ref_pad, (11,11,3)).squeeze()

        # mu_src.shape = (512, 512, 3)
        # mu_src_win_ext.shape = (512, 512, 11, 11, 3)
        mu_src_win = np.expand_dims(mu_src, (2,3))
        mu_src_win_ext = np.tile(mu_src_win, (1, 1, 11, 11, 1))
        mu_ref_win = np.expand_dims(mu_ref, (2,3))
        mu_ref_win_ext = np.tile(mu_ref_win, (1, 1, 11, 11, 1))

        kernel_gaus_3d = np.concatenate((np.expand_dims(kernel_gaus, 2), np.expand_dims(kernel_gaus, 2), np.expand_dims(kernel_gaus, 2)), 2)

        kernel_gaus_3d = np.expand_dims(kernel_gaus_3d, (0,1))
        kernel_gaus_3d_ext = np.tile(kernel_gaus_3d, (512, 512, 1, 1, 1))

        src_pad_ext_norm = src_pad_ext - mu_src_win_ext
        ref_pad_ext_norm = ref_pad_ext - mu_ref_win_ext
        sig_src = np.sum(kernel_gaus_3d_ext * src_pad_ext_norm ** 2, axis=(2,3)) ** 0.5
        sig_ref = np.sum(kernel_gaus_3d_ext * ref_pad_ext_norm ** 2, axis=(2,3)) ** 0.5
        cov = np.sum(kernel_gaus_3d_ext * src_pad_ext_norm * ref_pad_ext_norm, axis=(2,3))
 
        l = (2 * mu_src * mu_ref + c1) / (mu_src ** 2 + mu_ref ** 2 + c1)
        c = (2 * sig_src * sig_ref + c2) / (sig_src ** 2 + sig_ref ** 2 + c2)
        s = (cov + c3) / (sig_src * sig_ref + c3)

        ssim_local = l ** alp * c ** bet * s ** gam
        mssim = np.sum(ssim_local, axis=(0,1)) / M
        mssim = mssim[0] * w_r + mssim[1] * w_g + mssim[2] * w_b

        return l, c, s
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def gradient_img(img):
        img_h, img_w, img_c = img.shape
        edge_mag = np.zeros((img_h, img_w, 0))
        for c in range(3):
            grad_x = cv2.Sobel(img[:,:,c], cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(img[:,:,c], cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            sobel_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            edge_mag = np.concatenate((edge_mag, np.expand_dims(sobel_mag, 2)), 2)
        return edge_mag
    
    # ------------------------------------------------------------------------------------------------------------------
    # R1 regions: preserved edge pixel region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R1_region(ssim_loc, src_edge, ref_edge, T1):
        #ssim_r1 = ()
        R1 = ()
        for c in range(3):
            R1_c = ssim_loc[:,:,c][(src_edge[:,:,c] > T1) & (ref_edge[:,:,c] > T1)]
            # if c == 0:
            #     print(R1_c.shape)
            #ssim_r1_c = np.sum(R1_c) / R1_c.shape[0]
            #ssim_r1 = ssim_r1 + (ssim_r1_c,)
            R1 = R1 + (R1_c,)
        #return ssim_r1
        return R1


    # ------------------------------------------------------------------------------------------------------------------
    # R2 regions: changed edge pixel region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R2_region(ssim_loc, src_edge, ref_edge, T1):
        #ssim_r2 = ()
        R2 = ()
        for c in range(3):
            R2_c = ssim_loc[:,:,c][(src_edge[:,:,c] > T1) & (ref_edge[:,:,c] <= T1) | 
                                   (ref_edge[:,:,c] > T1) & (src_edge[:,:,c] <= T1)]
            # if c == 0:
            #     print(R2_c.shape)
            #ssim_r2_c = np.sum(R2_c) / R2_c.shape[0]
            #ssim_r2 = ssim_r2 + (ssim_r2_c,)
            R2 = R2 + (R2_c,)
        #return ssim_r2
        return R2

    # ------------------------------------------------------------------------------------------------------------------
    # R3 regions: Smooth region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R3_region(ssim_loc, src_edge, ref_edge, T1, T2):
        #ssim_r3 = ()
        R3 = ()
        for c in range(3):
            R3_c = ssim_loc[:,:,c][(src_edge[:,:,c] < T2) & (ref_edge[:,:,c] > T1)]
            # if c == 0:
            #     print(R3_c.shape)
            #ssim_r3_c = np.sum(R3_c) / R3_c.shape[0]
            #ssim_r3 = ssim_r3 + (ssim_r3_c,)
            R3 = R3 + (R3_c,)
        #return ssim_r3
        return R3

    # ------------------------------------------------------------------------------------------------------------------
    # R4 regions: Rexture region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R4_region(ssim_loc, src_edge, ref_edge, T1, T2):
        #ssim_r4 = ()
        R4 = ()
        for c in range(3):
            # R4_c = ssim_loc[:,:,c][(src_edge[:,:,c] <= T1) & 
            #                        ((ref_edge[:,:,c] <= T1) | 
            #                        ((src_edge[:,:,c] > T1) & (src_edge[:,:,c] >= T2))) ]
            R4_c = ssim_loc[:,:,c][~((src_edge[:,:,c] > T1) & (ref_edge[:,:,c] > T1)) &
                                   ~((src_edge[:,:,c] > T1) & (ref_edge[:,:,c] <= T1) | (ref_edge[:,:,c] > T1) & (src_edge[:,:,c] <= T1)) &
                                   ~((src_edge[:,:,c] < T2) & (ref_edge[:,:,c] > T1))]
            # if c == 0:
            #     print(R4_c.shape)
            #ssim_r4_c = np.sum(R4_c) / R4_c.shape[0]
            #ssim_r4 = ssim_r4 + (ssim_r4_c,)
            R4 = R4 + (R4_c,)
        # print("Total: " + str(262144))
        # print("R4: " + str(189189))
        #return ssim_r4
        return R4
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref):
        src_img = src.get_raw()
        ref_img = ref.get_raw()

        alp = bet = gam = 1.0



        src_img = src.get_raw()
        ref_img = ref.get_raw()

        # apply Sobel filter for each channel
        src_edge_mag = IVEGSSIM.gradient_img(src_img)
        ref_edge_mag = IVEGSSIM.gradient_img(ref_img)

        l, _, _ = IVEGSSIM.ssim(src_img, ref_img)
        _, c, s = IVEGSSIM.ssim(np.log(src_edge_mag+1), np.log(ref_edge_mag+1))
        ssim_local = l ** alp * c ** bet * s ** gam

        T1 = 0.12 * np.max(src_edge_mag)
        T2 = 0.06 * np.max(src_edge_mag)

        
        R1 = IVEGSSIM.R1_region(ssim_local, src_edge_mag, ref_edge_mag, T1)
        R2 = IVEGSSIM.R2_region(ssim_local, src_edge_mag, ref_edge_mag, T1)
        R3 = IVEGSSIM.R3_region(ssim_local, src_edge_mag, ref_edge_mag, T1, T2)
        R4 = IVEGSSIM.R4_region(ssim_local, src_edge_mag, ref_edge_mag, T1, T2)

        # setting of weights
        #           w1      w2      w3      w4
        # if R1={}  0.00    0.50    0.25    0.25
        # if R2={}  0.50    0.00    0.25    0.25
        # else      0.25    0.25    0.25    0.25
        ivssim_val = np.zeros(3)
        w3 = w4 = 0.25
        for c in range(3):
            if R1[c].shape[0] == 0:
                 w1 = 0.0
                 w2 = 0.5
            elif R2[c].shape[0] == 0:
                 w1 = 0.5
                 w2 = 0.0
            else:
                 w1 = 0.25
                 w2 = 0.25
        
            if R1[c].shape[0] != 0.0:
                ivssim_val[c] += np.sum(R1[c]) / R1[c].shape[0] * w1
            if R2[c].shape[0] != 0.0:
                ivssim_val[c] += np.sum(R2[c]) / R2[c].shape[0] * w2
            if R3[c].shape[0] != 0.0:
                ivssim_val[c] += np.sum(R3[c]) / R3[c].shape[0] * w3
            if R4[c].shape[0] != 0.0:
                ivssim_val[c] += np.sum(R4[c]) / R4[c].shape[0] * w4

        w_r = w_g = w_b = 1.0/3.0
        mivssim = ivssim_val[0] * w_r + ivssim_val[1] * w_g + ivssim_val[2] * w_b

        return round(mivssim, 4)
    
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main2():
    file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "GLO"
    total_tests = 0
    eval_arr = []



    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        ssim = IVEGSSIM.apply(src, out)
        print(ssim)

        eval_arr.append(ssim)

        with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/ivegssim.txt","a") as file2:
            file2.writelines(str(round(ssim,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



        # calculate mean
    mean = sum(eval_arr) / len(eval_arr)

    # calculate std
    std = 0
    for t in eval_arr:
        std += math.pow(t-mean, 2)
    std /= len(eval_arr)
    std = math.sqrt(std)


    print("Averaged: " + str(round(mean,3)) + " +- " + str(round(std,3)))

    file1.close()

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "PDF"
    total_tests = 0
    eval_arr = []

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    max_processes = 20
    sem = Semaphore(max_processes)

    def task(i, return_dict, sem):
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        #print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        ssim = IVEGSSIM.apply(src, out)
        return_dict[i] = (ssim, str(round(ssim,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")
        sem.release()

    for line in file1.readlines():
        total_tests += 1

        sem.acquire() # acquire semaphore before starting a new process
        process = Process(target=task, args=(total_tests, return_dict, sem))
        jobs.append(process)
        process.start()

        print(total_tests)

        # if total_tests == 20:
        #     break

    for i, proc in enumerate(jobs):
        print(i)
        proc.join()

    print("DONE")

    for i, val in enumerate(return_dict.values()):
        print(i)
        eval_arr.append(val[0])
        with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/ivegssim.txt","a") as file2:
           file2.writelines(val[1])

    # calculate mean
    mean = sum(eval_arr) / len(eval_arr)

    # calculate std
    std = 0
    for t in eval_arr:
        std += math.pow(t-mean, 2)
    std /= len(eval_arr)
    std = math.sqrt(std)
    print("Averaged: " + str(round(mean,3)) + " +- " + str(round(std,3)))
    file1.close()

    exit()

    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        ssim = IVEGSSIM.apply(src, out)
        print(ssim)

        eval_arr.append(ssim)

        with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/ivegssim.txt","a") as file2:
            file2.writelines(str(round(ssim,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



        # calculate mean
    mean = sum(eval_arr) / len(eval_arr)

    # calculate std
    std = 0
    for t in eval_arr:
        std += math.pow(t-mean, 2)
    std /= len(eval_arr)
    std = math.sqrt(std)


    print("Averaged: " + str(round(mean,3)) + " +- " + str(round(std,3)))

    file1.close()

if __name__ == "__main__":
    main()
