function [result] = mgRecolourPixels(pix, A, Nv, ctrl)	
if(size(pix,2)==3)
    if exist('mex_mgRecolourParallel_1','file')
        [result] = mex_mgRecolourParallel_1(pix', A', Nv', ctrl');
    else
        message = ['Precompiled mgrecolourParallel module not found.\n' ...
            'If the corresponding MEX-functions exist, run the following command:\n' ...
            'mex -g  mex_mgRecolourParallel_1.cpp COMPFLAGS="/openmp $COMPFLAGS"'];
        message_id = 'MATLAB:MEXNotFound';
        error (message_id, message);
    end
else
    if exist('mex_mgRecolourParallel2d','file')
        [result] = mex_mgRecolourParallel2d(pix', A', Nv', ctrl');
    else
        message = ['Precompiled mgrecolourParallel module not found.\n' ...
            'If the corresponding MEX-functions exist, run the following command:\n' ...
            'mex -g  mex_mgRecolourParallel2d.cpp COMPFLAGS="/openmp $COMPFLAGS"'];
        message_id = 'MATLAB:MEXNotFound';
        error (message_id, message);
    end
        
    
end

