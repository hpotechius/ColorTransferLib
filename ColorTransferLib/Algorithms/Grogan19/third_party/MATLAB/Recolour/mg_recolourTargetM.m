function [ finalResult ] = mg_recolourTargetM( target, targetMask, paramB, paramW, ctrl_pts, colourSpace )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
A = ((imread(target)));
if(strcmp(colourSpace, 'LAB'))
    A = rgb2lab(A);
end
[fnrows, fncols, d] = size(A);
fullT = reshape(double(A),fnrows*fncols,3);
M = imread(targetMask);
nM = imresize(M, [fnrows, fncols]);
fullM = double(reshape(M(:,:,1),fnrows*fncols,1));
fullM = fullM./255;
newFrame = mg_transform_parallel_mask(paramB,paramW,fullM,fullT, ctrl_pts);%recolour target image in parallel
if(strcmp(colourSpace, 'LAB'))
    newFrame = 255*lab2rgb(newFrame);
end

finalResult = reshape(newFrame, [fnrows fncols 3]);
finalResult = uint8(finalResult);



