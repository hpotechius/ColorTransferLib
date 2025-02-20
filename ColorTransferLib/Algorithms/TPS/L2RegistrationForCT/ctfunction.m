%function [param, config] = ctfunction(paletteFile, targetFile, varargin);
function finalResult = ctfunction(he2, he1, varargin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ctfunction(paletteFile, targetFile, clusterFun, nColors,colourSpace )
% Computes the colour transfer result between a target image and palette
% image that are aligned, eg. pixels at the same location in both images
% are correspondences.
%
% paletteFile:  The name of the palette file to be processed. The colour
%               distribution of the palette file will be mapped to the target image. 
% targetFile:   The name of the target file to be processed.The colour
%               distribution of the palette file will be mapped to the target image.
% clusterFun:   'KMeans' or 'MVQ'(default). The clustering function used.
% nColors:      The number of clusters computed by the clustering
%               function. The default is 50.
% colourSpace:  The colour space in which the registration is performed. The default is 'RGB'. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if(nargin < 2)
    %Three input arguements must be given
    error('Two input arguements are required: palette image, target image.');
elseif(nargin == 2)
    %If a clustering function is not specified, use Matlab's MVQ fucntion.
    %If the number of clusters to be used is not specified, use 50. 
    clusterFun = 'MVQ';
    nColors = 50;
    colourSpace = 'RGB';
elseif(nargin == 3)
    clusterFun = varargin{1};
    nColors = 50;
    colourSpace = 'RGB';
elseif(nargin == 4)
    clusterFun = varargin{1};
    nColors = varargin{2};
    colourSpace = 'RGB';
elseif(nargin == 5)
    clusterFun = varargin{1};
    nColors = varargin{2};
    colourSpace = varargin{3};
end
addpath(genpath('../L2RegistrationForCT'));

close all;

%read target image and reshape
%he1 = imread(targetFile);
if(strcmp(colourSpace, 'CIELab') && strcmp(clusterFun, 'KMeans'))
        he1	= rgb2lab(he1);
elseif(strcmp(colourSpace, 'CIELab') && strcmp(clusterFun, 'MVQ'))
    %disp('MVQ clustering cannot be applied in CIELAB space in this implementation. Using RGB space instead.');
    colourSpace = 'RGB';
end
dfull1 = double(he1);
fnrows = size(dfull1,1);
fncols = size(dfull1,2);
full_transform = reshape(dfull1,fnrows*fncols,3);

%read palette image
%he2 = imread(paletteFile);
if(strcmp(colourSpace, 'CIELab'))
        he2	= rgb2lab(he2);
end

%cluster target and palette image to get most dominant colours
%disp('Clustering of palette and target image started...');
switch(clusterFun)
    case 'KMeans'
        X = mg_applyKMeans(he1,nColors);
        Y = mg_applyKMeans(he2,nColors);
    case 'MVQ'
        X = mg_quantImage(he1, nColors);
        Y = mg_quantImage(he2, nColors);
end
%disp('Clustering of palette and target image finished.');

%initialise some parameters used to control the registration 
[config] = mg_initialize_config(X,Y, colourSpace);
%disp('Registration of colours started...');
%register the colours using an annealling scheme, estimate a TPS transformation 
for i = 1:config.AnnSteps
    config.iter = (config.AnnSteps-i+1);
    [param, transformed_model, history, config] = gmmreg_rbf_L2(config);
    if(i ~= config.AnnSteps)
        config.scale = .5*config.scale;
    end
end
%disp('Registration of colours finished.');

%apply the colour transformation to the target image to get the result
%image
%disp('Applying colour transfer to target image...');
fullTransform = mg_transform_tps_parallel(param, full_transform, config.ctrl_pts);
%disp('Finished.');
if(strcmp(colourSpace, 'CIELab'))
       fullTransform = lab2rgb(fullTransform);
       ind1 = find(fullTransform < 0);
       fullTransform(ind1) = 0;
       ind2 = find(fullTransform > 1);
       fullTransform(ind2) = 1;
       fullTransform = 255*fullTransform;
end

finalResult = reshape(fullTransform, [fnrows fncols 3]);
finalResult = uint8(finalResult);
%imshow(finalResult);

%sve the result
%disp('Saving the result in Results/result.png');
%imwrite(finalResult, 'Results/nocorrresult.png', 'png' );

end

