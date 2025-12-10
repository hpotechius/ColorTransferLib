% Der Name des relativen Pfades
%relativePath = 'util/mypath.mat';

% Gibt den vollständigen Pfad der aktuellen Datei zurück
%currentFilePath = mfilename('fullpath');

% Extrahiere nur den Ordner (ohne Dateiname)
%[currentFolder, ~, ~] = fileparts(currentFilePath);

% Drucke den Ordnerpfad
%disp(['Der aktuelle Pfad ist: ', currentFolder]);

% Kombiniere den aktuellen Pfad mit dem relativen Pfad
%fullPath = fullfile(currentFolder, relativePath);

pathroot = pwd;
%save -mat fullPath pathroot
save -mat util/mypath.mat pathroot
% save -mat /home/potechius/Code/ColorTransferLib/ColorTransferLib/Evaluation/VSI/gbvs/util/mypath.mat pathroot
addpath(genpath( pathroot ), '-begin');
savepath