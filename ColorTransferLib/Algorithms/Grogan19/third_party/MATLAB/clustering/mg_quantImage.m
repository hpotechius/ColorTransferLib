function [X] = mg_quantImage( RGB, n)
%apply matlab's rgb2ind to the RGB values given in 'RGB' to find n colors.
%[im,quant] = rgb2ind(RGB, n);
[im,quant] = rgb2ind(RGB);
X = unique(255*quant, 'rows'); 

end

