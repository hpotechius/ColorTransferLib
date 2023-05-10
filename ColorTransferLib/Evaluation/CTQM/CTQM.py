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
#import pysaliency
from multiprocessing import Process, Pool, Manager, Semaphore

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Color Transfer Quality Measure (CTQM)
# ...

# Source: Novel multi-color transfer algorithms and quality measure
#
# Range [...]
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CTQM:
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
        kernel_gaus = CTQM.iso2Dgauss()

        src_img = src
        ref_img = ref

        hi, wi = src_img.shape

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
        
        mu_src = cv2.filter2D(src_img, -1, kernel_gaus)
        mu_ref = cv2.filter2D(ref_img, -1, kernel_gaus)

        src_pad = np.pad(src_img, ((5, 5), (5, 5)), 'reflect')
        ref_pad = np.pad(ref_img, ((5, 5), (5, 5)), 'reflect')

        sig_src = np.zeros_like(src_img)
        sig_ref = np.zeros_like(ref_img)
        cov = np.zeros_like(src_img)

        # Version 3
        # src_pad_ext.shape = (512, 512, 1, 11, 11, 3)
        src_pad_ext = np.lib.stride_tricks.sliding_window_view(src_pad, (11,11)).squeeze()
        ref_pad_ext = np.lib.stride_tricks.sliding_window_view(ref_pad, (11,11)).squeeze()

        # mu_src.shape = (512, 512)
        # mu_src_win_ext.shape = (512, 512, 11, 11)
        mu_src_win = np.expand_dims(mu_src, (2,3))
        mu_src_win_ext = np.tile(mu_src_win, (1, 1, 11, 11))
        mu_ref_win = np.expand_dims(mu_ref, (2,3))
        mu_ref_win_ext = np.tile(mu_ref_win, (1, 1, 11, 11))

        kernel_gaus = np.expand_dims(kernel_gaus, (0,1))
        kernel_gaus_ext = np.tile(kernel_gaus, (hi, wi, 1, 1))

        src_pad_ext_norm = src_pad_ext - mu_src_win_ext
        ref_pad_ext_norm = ref_pad_ext - mu_ref_win_ext
        sig_src = np.sum(kernel_gaus_ext * src_pad_ext_norm ** 2, axis=(2,3)) ** 0.5
        sig_ref = np.sum(kernel_gaus_ext * ref_pad_ext_norm ** 2, axis=(2,3)) ** 0.5
        cov = np.sum(kernel_gaus_ext * src_pad_ext_norm * ref_pad_ext_norm, axis=(2,3))
 
        l = (2 * mu_src * mu_ref + c1) / (mu_src ** 2 + mu_ref ** 2 + c1)
        c = (2 * sig_src * sig_ref + c2) / (sig_src ** 2 + sig_ref ** 2 + c2)
        s = (cov + c3) / (sig_src * sig_ref + c3)

        #ssim_local = l ** alp * c ** bet * s ** gam
        #mssim = np.sum(ssim_local, axis=(0,1)) / M

        return l, c, s
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def gradient_img(img):
        grad_x = cv2.Sobel(img[:,:], cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img[:,:], cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        sobel_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return sobel_mag
    
    # ------------------------------------------------------------------------------------------------------------------
    # R1 regions: preserved edge pixel region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R1_region(ssim_loc, src_edge, ref_edge, T1):
        R1 = ssim_loc[:,:][(src_edge[:,:] > T1) & (ref_edge[:,:] > T1)]
        return R1


    # ------------------------------------------------------------------------------------------------------------------
    # R2 regions: changed edge pixel region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R2_region(ssim_loc, src_edge, ref_edge, T1):
        R2 = ssim_loc[:,:][(src_edge[:,:] > T1) & (ref_edge[:,:] <= T1) | 
                                (ref_edge[:,:] > T1) & (src_edge[:,:] <= T1)]
        return R2

    # ------------------------------------------------------------------------------------------------------------------
    # R3 regions: Smooth region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R3_region(ssim_loc, src_edge, ref_edge, T1, T2):
        R3 = ssim_loc[:,:][(src_edge[:,:] < T2) & (ref_edge[:,:] > T1)]
        return R3

    # ------------------------------------------------------------------------------------------------------------------
    # R4 regions: Rexture region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R4_region(ssim_loc, src_edge, ref_edge, T1, T2):
        R4 = ssim_loc[:,:][~((src_edge[:,:] > T1) & (ref_edge[:,:] > T1)) &
                                ~((src_edge[:,:] > T1) & (ref_edge[:,:] <= T1) | (ref_edge[:,:] > T1) & (src_edge[:,:] <= T1)) &
                                ~((src_edge[:,:] < T2) & (ref_edge[:,:] > T1))]
        return R4
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def ivegssim(src, ref):
        src_img = src
        ref_img = ref

        alp = bet = gam = 1.0

        # apply Sobel filter for each channel
        src_edge_mag = CTQM.gradient_img(src_img)
        ref_edge_mag = CTQM.gradient_img(ref_img)

        l, _, _ = CTQM.ssim(src_img, ref_img)
        _, c, s = CTQM.ssim(np.log(src_edge_mag+1), np.log(ref_edge_mag+1))
        ssim_local = l ** alp * c ** bet * s ** gam

        T1 = 0.12 * np.max(src_edge_mag)
        T2 = 0.06 * np.max(src_edge_mag)

        
        R1 = CTQM.R1_region(ssim_local, src_edge_mag, ref_edge_mag, T1)
        R2 = CTQM.R2_region(ssim_local, src_edge_mag, ref_edge_mag, T1)
        R3 = CTQM.R3_region(ssim_local, src_edge_mag, ref_edge_mag, T1, T2)
        R4 = CTQM.R4_region(ssim_local, src_edge_mag, ref_edge_mag, T1, T2)

        # setting of weights
        #           w1      w2      w3      w4
        # if R1={}  0.00    0.50    0.25    0.25
        # if R2={}  0.50    0.00    0.25    0.25
        # else      0.25    0.25    0.25    0.25
        ivssim_val = 0.0
        w3 = w4 = 0.25

        if R1.shape[0] == 0:
            w1 = 0.0
            w2 = 0.5
        elif R2.shape[0] == 0:
            w1 = 0.5
            w2 = 0.0
        else:
            w1 = 0.25
            w2 = 0.25
    
        if R1.shape[0] != 0.0:
            ivssim_val += np.sum(R1) / R1.shape[0] * w1
        if R2.shape[0] != 0.0:
            ivssim_val += np.sum(R2) / R2.shape[0] * w2
        if R3.shape[0] != 0.0:
            ivssim_val += np.sum(R3) / R3.shape[0] * w3
        if R4.shape[0] != 0.0:
            ivssim_val += np.sum(R4) / R4.shape[0] * w4
        
        return ivssim_val
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rgb2rgyb(img):
        rg = img[:,:,0] - img[:,:,1]
        yb = 0.5 * (img[:,:,0] + img[:,:,1]) - img[:,:,2]
        return rg, yb
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def colorfulness(img):
        rg, yb = CTQM.rgb2rgyb(img * 255)

        mu_rg = np.mean(rg)
        mu_yb = np.mean(yb)

        sig_rg = np.std(rg)
        sig_yb = np.std(yb)

        rg_val = np.log(sig_rg ** 2 / np.abs(mu_rg) ** 0.2)
        yb_val = np.log(sig_yb ** 2 / np.abs(mu_yb) ** 0.2)

        cf = 0.2 * rg_val * yb_val

        return cf
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def colormapsimilarity(ref, out):
        # round a and b channel to integers
        ref = np.around(ref).astype(np.int32)
        out = np.around(out).astype(np.int32)

        # reshape from (512, 512, 2) to (262144, 2)
        ref = ref.reshape(ref.shape[0] * ref.shape[1], ref.shape[2])
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])

        # get normalized 2D histogram of a and b channel
        bins = [125, 125]
        #print(ref.shape)
        ref_histo = np.asarray(np.histogramdd(ref, bins)[0])
        out_histo = np.asarray(np.histogramdd(out, bins)[0])

        ref_histo /= np.sum(ref_histo)
        out_histo /= np.sum(out_histo)

        cms = -np.sum(np.abs(out_histo-ref_histo))

        return cms


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = args[0]
        ref = args[1]
        out = args[2]
        # RGB to CIELab
        src_lab = cv2.cvtColor(src.get_raw(), cv2.COLOR_RGB2LAB)
        ref_lab = cv2.cvtColor(ref.get_raw(), cv2.COLOR_RGB2LAB)
        out_lab = cv2.cvtColor(out.get_raw(), cv2.COLOR_RGB2LAB)

        # Caluclate 4-EGSSIM of L channel
        # multiple with 2.55 to get range [0, 255] instead of [0, 100]
        ivegssim_val = CTQM.ivegssim(out_lab[:,:,0], src_lab[:,:,0])

        # Calculate Colorfulness
        cf_val = CTQM.colorfulness(out.get_raw())

        # Calculate Color Map Similarity
        cms_val = CTQM.colormapsimilarity(ref_lab[:,:,1:3], out_lab[:,:,1:3])

        # Calculate CTQM
        wo = 1.0
        ws = 100.0
        wm = 1.0

        ctqm_val = wo * cf_val + wm * cms_val + ws *ivegssim_val

        return round(ctqm_val, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    with open("/media/potechius/Active_Disk/Tests/MetricEvaluation/GLO/ctqm.txt","r") as file2:
        cc = 0
        summ = 0
        for line in file2.readlines():
            tim = float(line.strip().split(" ")[0])
            if math.isinf(tim) or math.isnan(tim):
                continue
            summ += tim
            cc += 1
        summ /= cc
        print(cc)
        print(summ)
    exit()    
    """
    outfile_name = "/media/potechius/Backup_00/Tests/MetricEvaluation/"+"GLO"+"/512_interior-01_dithering-4__to__512_nature-09_dithering-4.png"
    img_tri = cv2.imread(outfile_name)
    src_img = img_tri[:,:512,:]
    ref_img = img_tri[:,512:1024,:]
    out_img = img_tri[:,1024:,:]

    src = Image(array=src_img)
    ref = Image(array=ref_img)
    out = Image(array=out_img)
    ssim = CTQM.apply(src, ref, out)
    print(ssim)
    exit()
    """

    file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "BCC"
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
        ssim = CTQM.apply(src, ref, out)
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
        with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/ctqm.txt","a") as file2:
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

    file1 = open("/media/potechius/Active_Disk/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "GLO"
    total_tests = 0
    eval_arr = []
    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/potechius/Active_Disk/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        ssim = CTQM.apply(src, ref, out)

        eval_arr.append(ssim)

        with open("/media/potechius/Active_Disk/Tests/MetricEvaluation/"+ALG+"/ctqm.txt","a") as file2:
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
