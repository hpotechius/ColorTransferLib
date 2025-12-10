function [ctrl_pts] = set_ctrl_pts(M,S, interval, d, colourSpace)
%%=====================================================================
%% Project:   Point Set Registration using Gaussian Mixture Model
%% Module:    $RCSfile: set_ctrl_pts.m,v $
%% Language:  MATLAB
%% Author:    $Author: bing.jian $
%% Date:      $Date: 2008-11-13 21:34:29 +0000 (Thu, 13 Nov 2008) $
%% Version:   $Revision: 109 $
%% i this the functions
%%=====================================================================

if(strcmp(colourSpace, 'CIELab'))

    x_min = 0;
    x_max = 100;
    y_min = -100;
    y_max = 100;
        if(d ==2)
            [x,y] = ndgrid(linspace(x_min,x_max,interval), linspace(y_min,y_max,interval));
            n_pts = interval*interval;
            ctrl_pts = [reshape(x,n_pts,1) reshape(y,n_pts,1)];    
        else
            z_min = -110;
            z_max = 100;
            [x,y,z] = ndgrid(linspace(x_min,x_max,interval), linspace(y_min,y_max,interval),linspace(z_min,z_max,interval));
            n_pts = interval*interval*interval;
            ctrl_pts = [reshape(x,n_pts,1) reshape(y,n_pts,1) reshape(z,n_pts,1)];    
        end
else
    x_min = 0;
    x_max = 255;
    y_min = 0;
    y_max = 255;
        if(d ==2)
            [x,y] = ndgrid(linspace(x_min,x_max,interval), linspace(y_min,y_max,interval));
            n_pts = interval*interval;
            ctrl_pts = [reshape(x,n_pts,1) reshape(y,n_pts,1)];    
        else
            z_min = 0;
            z_max = 255;
            [x,y,z] = ndgrid(linspace(x_min,x_max,interval), linspace(y_min,y_max,interval),linspace(z_min,z_max,interval));
            n_pts = interval*interval*interval;
            ctrl_pts = [reshape(x,n_pts,1) reshape(y,n_pts,1) reshape(z,n_pts,1)];    
        end
 end
 end