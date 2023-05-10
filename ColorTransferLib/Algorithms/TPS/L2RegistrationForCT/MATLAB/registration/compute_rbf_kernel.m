%% compute the kernel and basis matrix for rbf
function [U, K] = compute_rbf_kernel(ctrl_pts,landmarks, kernel, kernelParam)
%%=====================================================================
%% $RCSfile: compute_rbf_kernel.m,v $
%% $Author: mairead grogan $

%%=====================================================================
[n,d] = size(ctrl_pts);
[m,d] = size(landmarks);
U = zeros(m,n);
K = zeros(n,n);

for i=1:m
    for j=1:n
        r = norm(landmarks(i,1:3) - ctrl_pts(j,1:3));
        U(i,j) =  mg_apply_kernel(r, kernel, kernelParam);
    end
end


for i=1:n
    for j=1:n
        r = norm(ctrl_pts(i,1:3) - ctrl_pts(j,1:3));
        K(i,j) =   mg_apply_kernel(r, kernel, kernelParam);
    end
end





