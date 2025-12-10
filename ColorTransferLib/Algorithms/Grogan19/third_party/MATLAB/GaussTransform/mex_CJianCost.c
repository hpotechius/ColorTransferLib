/*%%=====================================================================
%% Project:   Pointset Registration using Gaussian Mixture Model
%% Module:    $RCSfile: mex_GaussTransform.c,v $
%% Language:  C
%% Author:    $Author: bing.jian $
%% Date:      $Date: 2008-11-13 21:34:29 +0000 (Thu, 13 Nov 2008) $
%% Version:   $Revision: 109 $
%%=====================================================================*/

#include "mex.h"

double CJianCost(double* A, double* B, double* CorrA, double* CorrB, int m, int n, int dim, int numCorr, double scale1 , double* grad);//, double* grad);
void mexFunction(int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
    /* Declare variables */ 
    int m, n, dim, numCorr; 
    double *A, *B, *result, scale1, *grad;
	double *CorrA, *CorrB;
    
    /* Check for proper number of input and output arguments */    
    if (nrhs != 5) {
	mexErrMsgTxt("Five input arguments required.");
    } 
    if (nlhs > 2){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!(mxIsDouble(prhs[0]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    if (!(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    if (!(mxIsDouble(prhs[2]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    
    /* Get the number of elements in the input argument */
    /* elements=mxGetNumberOfElements(prhs[0]); */
    /* Get the data */
    A = (double *)mxGetPr(prhs[0]);
    B = (double *)mxGetPr(prhs[1]);
	CorrA = (double *)mxGetPr(prhs[2]);
	CorrB = (double *)mxGetPr(prhs[3]);
    scale1 = mxGetScalar(prhs[4]);
  	/* Get the dimensions of the matrix input A&B. */
  	m = mxGetN(prhs[0]);//Get the number of columns in the array A
  	n = mxGetN(prhs[1]);
  	dim = mxGetM(prhs[0]);
	numCorr = mxGetN(prhs[2]);
  	if (mxGetM(prhs[1])!=dim)
  	{
  		mexErrMsgTxt("The two input point sets should have same dimension.");
  	}
    /* Allocate the space for the return argument */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(dim,m,mxREAL);
    result = mxGetPr(plhs[0]);
    grad = mxGetPr(plhs[1]);
    *result = CJianCost(A, B,CorrA, CorrB, m, n, dim, numCorr, scale1, grad);
    
}

