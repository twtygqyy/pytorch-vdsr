clear;close all;
%% settings
folder = 'Set5';

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];

scale = [2, 3, 4];

for i = 1 : length(filepaths)
    im_gt = imread(fullfile(folder,filepaths(i).name));
    for s = 1 : length(scale) 
        im_gt = modcrop(im_gt, scale(s));
        im_gt = double(im_gt);
        im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
        im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
        im_l_ycbcr = imresize(im_gt_ycbcr,1/scale(s),'bicubic');
        im_b_ycbcr = imresize(im_l_ycbcr,scale(s),'bicubic');
        im_l_y = im_l_ycbcr(:,:,1) * 255.0;
        im_l = ycbcr2rgb(im_l_ycbcr) * 255.0;
        im_b_y = im_b_ycbcr(:,:,1) * 255.0;
        im_b = ycbcr2rgb(im_b_ycbcr) * 255.0;
        last = length(filepaths(i).name)-4;
        filename = sprintf('Set5_mat/%s_x%s.mat',filepaths(i).name(1 : last),num2str(scale(s)));
        save(filename, 'im_gt_y', 'im_b_y', 'im_gt', 'im_b', 'im_l_ycbcr', 'im_l_y', 'im_l');
    end
end
