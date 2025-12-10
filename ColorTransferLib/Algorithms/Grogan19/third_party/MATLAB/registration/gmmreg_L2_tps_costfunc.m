function [energy, grad] = gmmreg_L2_tps_costfunc(param, init_affine, basis,  kernel, scene, scale, alpha, beta, n, d)
%%=====================================================================
%% This function is based on gmmreg_L2_tps_costfunc, written by Jian et al and available for download here:https://github.com/bing-jian/gmmreg
%% these steps are detailed further in 'Robust Point Set Registration
%% Using Gaussian Mixture Models'
%%=====================================================================

if isempty(init_affine)
    %% if init_affine is given as [ ], then it means the affine matrix is 
    %% part of parameter and will be updated during optimization as well.
    %% In this case, the length of parameter should be n*d
    affine_param = reshape(param(1:d*(d+1)),d,d+1);
    affine_param = affine_param';
    tps_param = reshape(param(d*(d+1)+1:d*n),d,n-d-1);
    tps_param = tps_param';
else
    %% if a non-empty init_affine is given, then it will be treated as
    %% a fixed affine matrix.
    %% In this case, the length of parameter should be (n-d-1)*d
    tps_param = reshape(param(1:d*n-d*(d+1)),d,n-d-1);
    tps_param = tps_param';
    affine_param = reshape(init_affine,d,d+1);
    affine_param = affine_param';
end
after_tps = basis*[affine_param;tps_param];
bending = trace(tps_param'*kernel*tps_param);
[energy,grad] = general_costfunc(after_tps, scene, scale);
energy = alpha*energy + beta * bending;
grad = alpha*basis'*grad;
grad(d+2:n,:) = grad(d+2:n,:) + 2*beta*kernel*tps_param;
if isempty(init_affine) 
    %% In this case, the length of gradient should be n*d    
    grad = grad';
    grad = reshape(grad,1,d*n);
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
