function [warped_pts, bending_energy] = mg_transform_tps_parallel(param, landmarks, ctrl_pts)
%%=====================================================================
%Function created by Mairead Grogan October 2015
% Computes pixel recolouring in parallel for TPS transformation
%%=====================================================================
    [n,d] = size(ctrl_pts);
    param = reshape([param],d,n); param = param'; 
    Pn = [ones(n,1) ctrl_pts];
    PP = null(Pn'); 
    Nv = PP*param((d+2):end, :);
    %param(1:(d+1), 1:d)
    warped_pts = mgRecolourPixels(landmarks, param(1:(d+1), 1:d), Nv, ctrl_pts);
    warped_pts = warped_pts';


