%%% Prerequisites
% You need to get the cpp file edgesNmsMex.cpp from https://raw.githubusercontent.com/pdollar/edges/master/private/edgesNmsMex.cpp
% and compile it in Matlab: mex edgesNmsMex.cpp
% You also need to download and install Piotr's Computer Vision Matlab Toolbox:  https://pdollar.github.io/toolbox/

%%% parameters
% hed_mat_dir: the hed mat file directory (the output of 'batch_hed.py')
% edge_dir: the output HED edges directory
% image_width: resize the edge map to [image_width, image_width] 
% threshold: threshold for image binarization (default 25.0/255.0)
% small_edge: remove small edges (default 5)

function [] = PostprocessHED(hed_mat_dir, edge_dir, image_width, threshold, small_edge)

if ~exist(edge_dir, 'dir')
    mkdir(edge_dir);
end
fileList = dir(fullfile(hed_mat_dir, '*.mat'));
nFiles = numel(fileList);
fprintf('find %d mat files\n', nFiles);

for n = 1 : nFiles
    if mod(n, 1000) == 0
        fprintf('process %d/%d images\n', n, nFiles);
    end
    fileName = fileList(n).name;
    filePath = fullfile(hed_mat_dir, fileName);
    jpgName = strrep(fileName, '.mat', '.jpg');
    edge_path = fullfile(edge_dir, jpgName);
    
    if ~exist(edge_path, 'file')
        E = GetEdge(filePath);
        E = imresize(E,[image_width,image_width]);
        E_simple = SimpleEdge(E, threshold, small_edge);
        E_simple = uint8(E_simple*255);
        imwrite(E_simple, edge_path, 'Quality',100);
    end
end
end




function [E] = GetEdge(filePath)
load(filePath);
E = 1-predict;
end

function [E4] = SimpleEdge(E, threshold, small_edge)
if nargin <= 1
    threshold = 25.0/255.0;
end

if nargin <= 2
    small_edge = 5;
end

if ndims(E) == 3
    E = E(:,:,1);
end

E1 = 1 - E;
E2 = EdgeNMS(E1);
E3 = double(E2>=max(eps,threshold));
E3 = bwmorph(E3,'thin',inf);
E4 = bwareaopen(E3, small_edge);
E4=1-E4;
end

function [E_nms] = EdgeNMS( E )
E=single(E);
[Ox,Oy] = gradient2(convTri(E,4));
[Oxx,~] = gradient2(Ox);
[Oxy,Oyy] = gradient2(Oy);
O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
E_nms = edgesNmsMex(E,O,1,5,1.01,1);
end
