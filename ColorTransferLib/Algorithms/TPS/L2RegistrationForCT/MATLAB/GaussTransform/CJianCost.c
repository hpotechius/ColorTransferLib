#define SQR(X)  ((X)*(X))

#include <math.h>
/* #include <stdio.h> */

/* PLEASE READ THIS HEADER, IMPORTANT INFORMATION INSIDE */
#include "memory_layout_note.h"

#ifdef WIN32
__declspec( dllexport )
#endif
double CJianCost(const double* A, const double* B,const double* CorrA, const double* CorrB, int m, int n, int dim, int numCorr, double scale1 , double* grad)
{
	int i,j,d, indi, indj; 
    int id, jd;
	double dist_ij, cross_term = 0;
    double cost_ij;
	for (i=0;i<m*dim;++i) grad[i] = 0;
	for (i=0;i<numCorr;++i)
	{
		indi = CorrA[i];
		indj = CorrB[i];
			dist_ij = 0;
			for (d=0;d<dim;++d)
			{
                id = indi*dim + d;
                jd = indj*dim + d;
				dist_ij = dist_ij + SQR( A[id] - B[jd]);
			}
            cost_ij = exp(-dist_ij/SQR(scale1));
			for (d=0;d<dim;++d){
                id = indi*dim + d;
                jd = indj*dim + d;
                grad[id] += -cost_ij*2*(A[id] - B[jd]);
            }
           
			cross_term += cost_ij;
	}
	for (i=0;i<m*dim;++i) {
		grad[i]/=(SQR(scale1));
	}

	return cross_term;
}