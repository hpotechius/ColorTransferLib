function outp = ct_apply(val);
#disp(val)
outp = 9.7
end
%% No correspondences [ctfunction(paletteFile, targetFile, clusterFun, nColors,colourSpace )]
%palette1 = 'data/images/kandinsky.jpg';
%target1 = 'data/images/northern_lights.jpg';
#disp(brot)


palette1 = 'data/images/hyacinth.jpg';
target1 = 'data/images/rose.jpg';
res = ctfunction(palette1,target1, 'KMeans', 50, 'RGB'); %this will apply colour transfer in RGB space with 50 Kmeans clusters selected from the target and palette images. (used to generate results in paper)
disp('Saving the result in Results/result.png');
imwrite(res, 'Results/nocorrresult.png', 'png' );

%end

% Some more samples showing how to set the parameters of ctfunction 
%ctfunction(palette1,target1, 'MVQ'); %this will apply colour transfer in RGB space using MVQ clustering, and 50 clusters as default (quicker than KMeans).
%ctfunction(palette1,target1, 'KMeans'); %this will apply colour transfer in RGB space using KMeans clustering, and 50 clusters as default.
%ctfunction(palette1,target1,'MVQ', 30); %this will apply colour transfer with MVQ clustering, computing 30 clusters in the traget and palette images.
%ctfunction(palette1,target1,'KMeans', 50, 'CIELab'); %this will apply colour transfer with KMeans clustering, computing 50 clusters in the target and palette images in CIELab space.

%% Correspondences [ctfunction_corr(paletteFile, targetFile, colourSpace, numCorr)]
%palette2 = 'reference_aligned.png';
%target2 = 'input.png';
%ctfunction_corr(palette2,target2);%this will apply colour transfer in RGB space using 5000 correspondences (assumed that the target and palette images are aligned.)

%Some more samples showing how to set the parameters of ctfunction_corr 
%ctfunction_corr(palette2,target2,'CIELab');
%ctfunction_corr(palette2,target2,'RGB', 50000); % set to 50000 to recreate results in paper.





%% display results
%close all;
%creensize = get(0,'ScreenSize');
%sz = [576, 1024];
%figure('Position', [ ceil((screensize(3)-sz(2))/2), ceil((screensize(4)-sz(1))/2), sz(2), sz(1)]);
%subplot('Position',[0.01  0.4850 0.3200 .47]); 
%imshow(imread(target1)); title('Target Image'); 

%subplot('Position',[0.3400  0.4850 0.3200 .47]); 
%imshow(imread(palette1)); title('Palette Image'); 

%subplot('Position',[0.67  0.4850 0.3200 .47]);
%imshow(imread('Results/nocorrresult.png')); title('Result after colour transfer');   

%subplot('Position',[0.01  0.01 0.3200 .47]); 
%imshow(imread(target2)); title('Target Image'); 

%subplot('Position',[0.3400  0.01 0.3200 .47]); 
%imshow(imread(palette2)); title('Palette Image'); 

%subplot('Position',[0.67  0.01 0.3200 .47]);
%imshow(imread('Results/corrresult.png')); title('Result after colour transfer'); 

