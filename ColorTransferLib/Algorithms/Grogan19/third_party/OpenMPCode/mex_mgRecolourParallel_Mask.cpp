#include "mex.h"
#include <omp.h>
#include <math.h>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	double *pix, *A1, *W1, *A2, *W2, *mask, *ctrl, *recolour;    
	int num_pix, num_ctrl;   

	if (nrhs != 7) {
	mexErrMsgTxt("Seven input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }

	if (!(mxIsDouble(prhs[0]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    if (!(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    if (!(mxIsDouble(prhs[2]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
	 if (!(mxIsDouble(prhs[3]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
	  if (!(mxIsDouble(prhs[4]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
	   if (!(mxIsDouble(prhs[5]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
	    if (!(mxIsDouble(prhs[6]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
		
    

	//inputs   
	pix = (double *)mxGetPr(prhs[0]);
	A1 = (double *)mxGetPr(prhs[1]);
	W1 = (double *)mxGetPr(prhs[2]);
	A2 = (double *)mxGetPr(prhs[3]);
	W2 = (double *)mxGetPr(prhs[4]);
	mask = (double *)mxGetPr(prhs[5]);
	ctrl = (double *)mxGetPr(prhs[6]);

	//figure out dimensions   
	num_pix = mxGetN(prhs[0]);
	num_ctrl = mxGetN(prhs[6]);  
	
	//associate outputs   
	plhs[0] = mxCreateDoubleMatrix(3,num_pix, mxREAL);    
	recolour = mxGetPr(plhs[0]);

	long int i, tmp_indx0,tmp_indx1,tmp_indx2;
	int j;
	double norm, knorm, k1, k2; 
	double W[375];
	double A[12];
	int nThreads = omp_get_max_threads();
	#pragma omp parallel for shared(pix, A1, W1, A2, W2, mask, ctrl, num_ctrl) private(j,norm, knorm, tmp_indx0, tmp_indx1, tmp_indx2, k1,k2, A, W)
	for(i = 0; i < num_pix; i++)
		{
			// Create new variables A that is k1*A1 + k2*A2
			k1 = 1 - mask[i];
			k2 = mask[i];
			A[0] = k1*A1[0] + k2*A2[0];		A[6] = k1*A1[6] + k2*A2[6];
			A[1] = k1*A1[1] + k2*A2[1];		A[7] = k1*A1[7] + k2*A2[7];
			A[2] = k1*A1[2] + k2*A2[2];		A[8] = k1*A1[8] + k2*A2[8];
			A[3] = k1*A1[3] + k2*A2[3];		A[9] = k1*A1[9] + k2*A2[9];
			A[4] = k1*A1[4] + k2*A2[4];		A[10] = k1*A1[10] + k2*A2[10];
			A[5] = k1*A1[5] + k2*A2[5];		A[11] = k1*A1[11] + k2*A2[11];

			// Do computation
			tmp_indx0 = 3*i;
			tmp_indx1 = 3*i+1;
			tmp_indx2 = 3*i+2;
			recolour[tmp_indx0] = A[0] + A[3]*pix[tmp_indx0] + A[6]*pix[tmp_indx1] + A[9]*pix[tmp_indx2]; // t+Ax1
			recolour[tmp_indx1] = A[1] + A[4]*pix[tmp_indx0] + A[7]*pix[tmp_indx1] + A[10]*pix[tmp_indx2]; //t+Ax2
			recolour[tmp_indx2] = A[2] + A[5]*pix[tmp_indx0] + A[8]*pix[tmp_indx1] + A[11]*pix[tmp_indx2]; //t+Ax3
		
			for(j = 0; j < num_ctrl; j++)
			{
				// Create new variables W that is k1*W1 + k2*W2
				W[j*3] =  k1*W1[j*3] + k2*W2[j*3];		
				W[(j*3) + 1] = k1*W1[(j*3) + 1] + k2*W2[(j*3) + 1];		
				W[(j*3) + 2] = k1*W1[(j*3) + 2] + k2*W2[(j*3) + 2];		

				norm = sqrt((pix[tmp_indx0] - ctrl[3*j])*(pix[tmp_indx0] - ctrl[3*j]) + (pix[tmp_indx1] - ctrl[(3*j)+1])*(pix[tmp_indx1] - ctrl[(3*j)+1]) + (pix[tmp_indx2] - ctrl[(3*j)+2])*(pix[tmp_indx2] - ctrl[(3*j)+2])); //|| Xi - cj||
				knorm = -norm;
				recolour[tmp_indx0] += knorm*(W[j*3]); //|| Xi - cj||*Wj1
				recolour[tmp_indx1] += knorm*(W[(j*3) + 1]); //|| Xi - cj||*Wj2
				recolour[tmp_indx2] += knorm*(W[(j*3) + 2]);//|| Xi - cj||*Wj3

				
			}
		}
	
	mexPrintf("nThreads = %i\n",nThreads);mexEvalString("drawnow");
}
