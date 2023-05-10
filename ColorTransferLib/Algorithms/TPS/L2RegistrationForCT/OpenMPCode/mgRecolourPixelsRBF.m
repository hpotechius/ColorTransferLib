function [result] = mgRecolourPixels(pix, A, w, ctrl, kernel, varargin)	
switch(kernel)
    case 'Gaussian' 
        b = varargin{1};
        if exist('mex_mgRecolourParallelGaussian','file')
            [result] = mex_mgRecolourParallelGaussian(pix', A', w', ctrl', b);
        else
            message = ['Precompiled mgrecolourParallel module not found.\n' ...
            'If the corresponding MEX-functions exist, run the following command:\n' ...
            'mex -g  mex_mgRecolourParallelGaussian.cpp COMPFLAGS="/openmp $COMPFLAGS"'];
            message_id = 'MATLAB:MEXNotFound';
            error (message_id, message);
        end
    
    case 'TPS'
        if exist('mex_mgRecolourParallelTPS','file')
            [result] = mex_mgRecolourParallelTPS(pix', A', w', ctrl');
        else
            message = ['Precompiled mgrecolourParallel module not found.\n' ...
            'If the corresponding MEX-functions exist, run the following command:\n' ...
           'mex -g  mex_mgRecolourParallelTPS.cpp COMPFLAGS="/openmp $COMPFLAGS"'];
            message_id = 'MATLAB:MEXNotFound';
            error (message_id, message);
        end
        
    case 'Multiquadric'
        b = varargin{1};
        if exist('mex_mgRecolourParallelMQ','file')
            [result] = mex_mgRecolourParallelMQ(pix', A', w', ctrl',b);
        else
            message = ['Precompiled mgrecolourParallel module not found.\n' ...
            'If the corresponding MEX-functions exist, run the following command:\n' ...
            'mex -g  mex_mgRecolourParallelMQ.cpp COMPFLAGS="/openmp $COMPFLAGS"'];
            message_id = 'MATLAB:MEXNotFound';
            error (message_id, message);
        end
    case 'Inversequadric'
        b = varargin{1};
        if exist('mex_mgRecolourParallelIQ','file')
            [result] = mex_mgRecolourParallelIQ(pix', A', w', ctrl',b);
        else
            message = ['Precompiled mgrecolourParallel module not found.\n' ...
            'If the corresponding MEX-functions exist, run the following command:\n' ...
             'mex -g  mex_mgRecolourParallelIQ.cpp COMPFLAGS="/openmp $COMPFLAGS"'];
            message_id = 'MATLAB:MEXNotFound';
            error (message_id, message);
        end
    case 'InverseMultiquadric'
        b = varargin{1};
        if exist('mex_mgRecolourParallelIM','file')
            [result] = mex_mgRecolourParallelIM(pix', A', w', ctrl',b);
        else
            message = ['Precompiled mgrecolourParallel module not found.\n' ...
            'If the corresponding MEX-functions exist, run the following command:\n' ...
            'mex -g  mex_mgRecolourParallelIM.cpp COMPFLAGS="/openmp $COMPFLAGS"'];
            message_id = 'MATLAB:MEXNotFound';
            error (message_id, message);
        end
end

