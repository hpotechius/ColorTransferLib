function [param, transformed_model, history, config] = gmmreg_rbf_L2(config)
%%=====================================================================
%% This function is based on gmmreg_L2, written by Jian et al and available for download here:https://github.com/bing-jian/gmmreg
%% these steps are detailed further in 'Robust Point Set Registration
%% Using Gaussian Mixture Models'
%%=====================================================================

history.x = [ ];
history.fval = [ ];
if nargin<1
    error('Usage: gmmreg_L2(config)');
end
[n,d] = size(config.model); % number of points in model set
if (d~=2)&&(d~=3)
    error('The current program only deals with 2D or 3D point sets.');
end

% Restrict the number of iterations for the first few annealling steps, if
% there are too many iterations when the scale is too large, a bad solution
% may be reached.
if(config.iter >1)
    options = optimset('GradObj','on', 'TolFun',1e-3, 'TolX',1e-3);
else
    options = optimset( 'GradObj','on', 'TolFun',1*10^(-10), 'TolX',1e-10);
end
%if(config.iter >1)
%    options = optimset( 'LargeScale','off','GradObj','on', 'TolFun',1e-3, 'TolX',1e-3, 'TolCon', 1e-3);
%else
%   options = optimset(  'LargeScale','off','GradObj','on', 'TolFun',1*10^(-10), 'TolX',1e-10, 'TolCon', 1e-10);
%end

%options = optimset(options, 'outputfcn',@outfun, 'PlotFcns', @optimplotfval );
options = optimset(options, 'MaxFunEvals', config.max_iter, 'MaxIter', config.max_iter);
options = optimset(options, 'GradObj', 'on');
options = optimset(options, 'Display', 'off');

switch lower(config.functionType)
    %when an rbf transformation is chosen
    case 'rbf'
        scene = config.scene;
        scale = config.scale;
        alpha = config.alpha;
        beta = config.beta;

        [n,d] = size(config.ctrl_pts);
        [m,d] = size(config.model);
        [U, K] = compute_rbf_kernel(config.ctrl_pts, config.model, config.kernel, config.kernelParam);
        Pm = [ones(m,1) config.model];
        Pn = [ones(n,1) config.ctrl_pts];
        %PP = null(Pn');  % or use qr(Pn)
        basis = [Pm U];
        %kernel = PP'*K*PP;

        init_rbf = config.init_rbf;  % it should always be of size d*(n-d-1)
        if isempty(config.init_affine)
            % for your convenience, [] implies default affine
            config.init_affine = repmat([zeros(1,d) 1],1,d);
        end
        if config.opt_affine % optimize both affine and tps
            init_affine = [ ];
            x0 = [config.init_affine init_rbf(1:end)];
        else % optimize tps only
            init_affine = config.init_affine;
            x0 = init_rbf(end+1-d*(n-d-1):end);
        end
       [ param,fval,exitflag,output,grad,hessian]  = fminunc(@(x)gmmreg_L2_rbf_costfunc(x, init_affine, basis, scene, scale, alpha, beta, n, d, K), x0,  options);
        transformed_model = mg_transform_rbf_parallel(param, config.model, config.ctrl_pts, config.kernel, config.kernelParam);
        if config.opt_affine
            config.init_rbf = param((d*(d+1))+1:end);
            config.init_affine = param(1:d*(d+1));
        else
            config.init_rbf = param;
        end
   %when an affine transformation is chosen
   case 'affine'
        x0 = config.init_param;
        param = fmincon(@gmmreg_L2_costfunc, x0, [ ],[ ],[ ],[ ], config.Lb, config.Ub, [ ], options, config);
        transformed_model = transform_pointset(config.model, 'affine3d', param);
        config.init_param = param;
    %when a tps transformation is chosen    
    case 'tps'
        scene = config.scene;
        scale = config.scale;
        alpha = config.alpha;
        beta = config.beta;

        [n,d] = size(config.ctrl_pts);
        [m,d] = size(config.model);
        [K,U] = compute_kernel(config.ctrl_pts, config.model);
        Pm = [ones(m,1) config.model];
        Pn = [ones(n,1) config.ctrl_pts];
        PP = null(Pn');  
        basis = [Pm U*PP];
        kernel = PP'*K*PP;

        init_tps = config.init_tps;  
        if isempty(config.init_affine)
            config.init_affine = repmat([zeros(1,d) 1],1,d);
        end
        if config.opt_affine % optimize both affine and tps
            init_affine = [ ];
            x0 = [config.init_affine init_tps(end+1-d*(n-d-1):end)];
        else % optimize tps only
            init_affine = config.init_affine;
            x0 = init_tps(end+1-d*(n-d-1):end);
        end
        
        %minimise the cost function 
        [param] = fminunc(@(x)gmmreg_L2_tps_costfunc(x, init_affine, basis, kernel, scene, scale, alpha, beta, n, d), x0,  options);
        
        transformed_model = mg_transform_tps_parallel(param, config.model, config.ctrl_pts);
        if config.opt_affine
            config.init_tps = param(end+1-d*(n-d-1):end);
            config.init_affine = param(1:d*(d+1));
            #config.init_affine
        else
            config.init_tps = param;
        end
 
end

    function stop = outfun(x,optimValues,state,varargin)
     stop = false;
     switch state
         case 'init'
             if config.display>0
               set(gca,'FontSize',16);
             end
         case 'iter'
               history.fval = [history.fval; optimValues.fval];
               history.x = [history.x; reshape(x,1,length(x))];
               if config.display>0
                   hold off
                   switch lower(config.functionType)
                       case 'tps'
                           transformed = transform_pointset(config.model, config.functionType, x, config.ctrl_pts,init_affine);
                       otherwise
                           transformed = transform_pointset(config.model, config.functionType, x);
                   end
                   dist = L2_distance(transformed,config.scene,config.scale);
                   DisplayPoints(transformed,config.scene,d);
                   title(sprintf('L2distance: %f',dist));
                   drawnow;
               end
         case 'done'
              %hold off
         otherwise
     end
    end


end



function [dist] = L2_distance(model, scene, scale)
    dist = GaussTransform(model,model,scale) + GaussTransform(scene,scene,scale) - 2*GaussTransform(model,scene,scale);
end




