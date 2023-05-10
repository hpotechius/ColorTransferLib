function [energy, grad] = gmmreg_L2_rbf_costfunc(param, init_affine, basis, scene, scale, alpha, beta, n, d, kernel)
%%=====================================================================
%% $RCSfile: gmmreg_L2_tps_costfunc.m,v $
%% $Author: bing.jian $
%% $Date: 2009-02-10 07:13:49 +0000 (Tue, 10 Feb 2009) $
%% $Revision: 121 $
%%=====================================================================

if isempty(init_affine)
    %% if init_affine is given as [ ], then it means the affine matrix is 
    %% part of parameter and will be updated during optimization as well.
    %% In this case, the length of parameter should be n*d
    affine_param = reshape(param(1:d*(d+1)),d,d+1);
    affine_param = affine_param';
    rbf_param = reshape(param(d*(d+1)+1:end),d,n);
    rbf_param = rbf_param';
else
    %% if a non-empty init_affine is given, then it will be treated as
    %% a fixed affine matrix.
    %% In this case, the length of parameter should be (n-d-1)*d
    rbf_param = reshape(param(1:d*n-d*(d+1)),d,n-d-1);
    rbf_param = rbf_param';
    affine_param = reshape(init_affine,d,d+1);
    affine_param = affine_param';
end
after_rbf = basis*[affine_param;rbf_param];
bending = trace(rbf_param'*kernel*rbf_param);
[energy,grad] = general_costfunc(after_rbf, scene, scale);
energy = alpha*energy + beta * bending;
grad = alpha*basis'*grad;
grad(d+2:(n+(d+1)),:) = grad(d+2:(n+(d+1)),:) + 2*beta*kernel*rbf_param;
if isempty(init_affine) 
    %% In this case, the length of gradient should be n*d    
    grad = grad';
    grad = reshape(grad,1,d*(n+ 4));
else 
    %% In this case, the length of parameter should be (n-d-1)*d    
    grad(1:d+1,:) = [ ];
    grad = grad';
    grad = reshape(grad,1,d*(n-d-1));
end

function [f, g] = general_costfunc(A, B, scale)
[f1, g1] = GaussTransform(A,A,scale);
[f2, g2] = GaussTransform(A,B,scale);
f =  f1 - 2*f2;
g = 2*g1 - 2*g2;


