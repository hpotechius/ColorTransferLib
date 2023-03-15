function [ cluster_center ] = mg_applyKMeans(image,nColors)
    %apply the k means algorithm to the image to find k colours that represent
    %it well 
    [n,m,d] = size(image); 
    image = imresize(image, [300 350]);
    %disp(size(image))
    ab = double(image);
    nrows = size(ab,1);
    ncols = size(ab,2);

    
    if(d == 3)
        ab = reshape(ab,nrows*ncols,d);
        [cluster_idx, cluster_center] = kmeans(ab, nColors, 'distance','sqEuclidean','MaxIter', 1000);

    %    [cluster_idx, cluster_center, clusters] = kmeans_reduced(ab, nColors);
    %    disp(["Amount ot Cluster = ", num2str(clusters)])
    %end

    %function [c_idx, c_center, clusters] = kmeans_reduced(im,colos)
    %    try            
    %        [c_idx, c_center] = kmeans(im, colos, 'distance','sqEuclidean','MaxIter', 1000);
    %        clusters = colos
    %    catch
    %        [c_idx, c_center, clusters] = kmeans_reduced(im,colos-1);
    %    end
    %end
end


