function [f,g] = gmmreg_L2_costfunc(param, config)
%%=====================================================================
%% $RCSfile: gmmreg_L2_costfunc.m,v $
%% $Author: bing.jian $
%% $Date: 2008-11-13 21:34:29 +0000 (Thu, 13 Nov 2008) $
%% $Revision: 109 $
%%=====================================================================
model = config.model;
scene = config.scene;
scale = config.scale;
[transformed_model] = transform_pointset(model, 'affine3d', param);

[f,grad] = general_costfunc(transformed_model, scene, scale);
grad = grad';
g(1) = sum(grad(1,:));
g(2) = sum(grad(2,:));
g(3) = sum(grad(3,:));
g(4:12) = reshape(grad*model,1,9);


function [f, g] = rigid_costfunc(A, B, scale)
[f, g] =  GaussTransform(A,B,scale);
f = -f; g = -g;


function [f, g] = general_costfunc(A, B, scale)
[f1, g1] = GaussTransform(A,A,scale);
[f2, g2] = GaussTransform(A,B,scale);
f =  f1 - 2*f2;
g = 2*g1 - 2*g2;