%%-------------------------------------------------------------------
%%
%% Author: Mairead Grogan
%% Date: April 2017
%% 
%%-------------------------------------------------------------------

This zip file contains the code for the L2 based colour transfer method 
described in 'L2 Divergence for robust colour transfer' published in Computer Vision and Image Understanding 2019. If using this code, please cite this paper. 


It contains code sourced from https://github.com/bing-jian/gmmreg  written by
Jian et al. in support of the paper 'Robust Point Set Registration Using Gaussian Mixture Models'. 
The web page for this paper can be found here: https://code.google.com/p/gmmreg/


The demo.m file is a script which shows how to run the colour transfer algorithm for both images with and without correspondeces. 
It implements the functions ctfunction.m (colour transfer applied to target and palette images without correspondences) and ctfunction_corr.m (colour transfer apllied to target and palette images with correspondences).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BEFORE RUNNING THE CODE: Building the mex files - 

1. Change the Matlab directory to the folder 'L2RegistrationForCT/MATLAB/GaussTransform' and run the script 'mg_initialiseMexFilesGT.m' to initialise the mex files. 
2. Change the Matlab directory to the folder 'L2RegistrationForCT/OpenMPCode' and run the script 'mg_initialiseMexFilesOMP.m' to initialise the mex files for recolouring that use open MP. 

This code uses Open MP within the mex files. Please ensure that you are using a C/C++ compiler which supports OpenMP. See: https://uk.mathworks.com/matlabcentral/answers/237411-can-i-make-use-of-openmp-in-my-matlab-mex-files

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FOLDER LAYOUT

L2RegistrationForCT/data/ - contains the image files for processing.

L2RegistrationForCT/Results/ - contains the result files after processing.

L2RegistrationForCT/OpenMPCode/ - contains the mex files and Matlab files used for recolouring the target image using the estimated colour transfer function. This step is parallelised using OpenMP. 

L2RegistrationForCT/MATLAB/  - Files in the 'MATLAB' directory are organised as follows:
	
	gmmreg_rbf_L2.m and gmmreg_L2_corr.m
	    The main entry into the MATLAB implementation fro images without and wih correspondences.

	mg_initialize_config.m and mg_initialize_config_corr.m
	    Generate the configuration struct used in gmmreg_rbf_L2.m and gmmreg_L2_corr.m.
		
	auxiliary/
	    Some supporting functions.
		
	clustering/
		Functions that apply the KMeans or MVQ clustering algorithms.
	
	GaussTransform/
	    MEX-files for implementing the GaussTransform with and without correspondences.

	registration/
	    Functions used in the Matlab implementation of the GMMReg algorithm, 
	    requiring 'GaussTransform' and the optimization toolbox.

	
	
		
		
Please address any queries to Mairead Grogan at mgrogan@tcd.ie
