function [ finalResult ] = mg_recolourTarget( target, param, ctrl_pts, colourSpace )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
A = ((imread(target)));
if(strcmp(colourSpace, 'LAB'))
    A = rgb2lab(A);
end
[fnrows, fncols, d] = size(A);
fullT = reshape(double(A),fnrows*fncols,3);
newFrame = mg_transform_tps_parallel(param, fullT, ctrl_pts);
if(strcmp(colourSpace, 'LAB'))
    newFrame = 255*lab2rgb(newFrame);
end
finalResult = reshape(newFrame, [fnrows fncols 3]);
finalResult = uint8(finalResult);