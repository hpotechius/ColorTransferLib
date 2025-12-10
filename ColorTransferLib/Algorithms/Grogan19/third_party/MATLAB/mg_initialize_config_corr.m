function [config] = mg_initialize_config_corr(model, scene,colourSpace, varargin)

config.model = model;
config.scene = scene;
% estimate the scale from the covariance matrix
[n,d] = size(model);
config.scale = power(det(model'*model/n), 1/(2^d));
config.display = 0;
config.init_param = [ ];
config.max_iter = 10000;
config.normalize = 0;
config.functionType = 'TPS';
config.AnnSteps = 1; %found that the best value it 1. 
config.scale = (2^(config.AnnSteps-1))*(config.scale);
switch lower(config.functionType)
    case 'tps'
        interval = 5;%found that the best value it 5. 
        config.ctrl_pts =  set_ctrl_pts(model, scene, interval, d, colourSpace);%set control points in a regular grid spanning the colour space 
        config.alpha = 1 - 0.003;
        config.beta = 0.003; % this value controls the strength of the regulsrisation term, we found 0.003 gives the best results when correspondences used
        config.opt_affine = 1;
        [n,d] = size(config.ctrl_pts); % number of points in model set
        config.init_tps = zeros(n-d-1,d);
        init_affine = repmat([zeros(1,d) 1],1,d);
        config.init_param = [init_affine zeros(1, d*n-d*(d+1))];
        config.init_affine = [ ];
end


