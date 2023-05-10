function [result] = mgRecolourPixels_Mask(pix, A1, Nv1, A2, Nv2, Mask,  ctrl)	
if(size(pix,2)==3)
    if exist('mex_mgRecolourParallel_Mask','file')
        [result] = mex_mgRecolourParallel_Mask(pix', A1', Nv1', A2', Nv2', Mask', ctrl');
    else
        message = ['Precompiled mgrecolourParallel_Mask module not found.\n' ...
            'If the corresponding MEX-functions exist, run the following command:\n' ...
            'mex -g  mex_mgRecolourParallel_Mask.cpp COMPFLAGS="/openmp $COMPFLAGS"'];
        message_id = 'MATLAB:MEXNotFound';
        error (message_id, message);
    end
else
    %To Do
        
    
end

