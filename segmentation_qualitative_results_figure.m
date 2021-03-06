% generates fig . 8 of the paper
% This code assumes that the several segmentation experiments in the
% saliency folder have been completed and all the results are stored in
% saliency/data/ferrari/

FIGS_PATH = 'genfigs/';

% Segmentation results figure. 
% We'll use 4 images and for each of them show
% the resized image, the ground truth segmentation mask, our result, result
% using saliency, result using deconvnet, the center seed baseline.

addpath('saliency');

NUM_IMAGES = 5;
ROOT = 'saliency/';
SALIENCY = 'sal-ks-101-alex';
DECONVNET = 'sal-dc-101-alex';
HYBRID = 'sal-am-101-alex';
BASELINE = 'sal-baseline3-101';

opts = struct();
opts.randomizeWeights = false;
opts.gpu = false;

opts.relus_to_change = 1:100;
opts.pools_to_change = 1:5;
opts.convs_to_change = 1:100;

opts.neuron_I = inf;
opts.neuron_J = inf;
opts.neuron_channel = inf;

opts.modelPath = 'models/imagenet-caffe-alex.mat';
opts.layer = 20;

pad_fn = @(x) padarray(x, [1,1], 1, 'both');

imdb = load('saliency/data/ferrari/imdb.mat');
%image_ids = randi(numel(imdb.images.name), [1, NUM_IMAGES]); % Randomly pick 5 images
image_ids = [ 3241, 3178, 1678, 2803, 732]; % There were generated by a call to the above line
image_names = cell(NUM_IMAGES, 1);
seg_names = cell(NUM_IMAGES, 1);
result_names = cell(NUM_IMAGES, 4);
viz = cell(NUM_IMAGES,1);
for i=1:NUM_IMAGES
    cur_img_name = imdb.images.name{image_ids(i)};
    image_names{i} = [ROOT, sprintf(imdb.paths.image, cur_img_name)];
    seg_names{i} = [ROOT, sprintf(imdb.paths.seg, cur_img_name, 1)];
    result_names{i,1} = [ROOT, 'data/ferrari/', SALIENCY, '/', cur_img_name, '.mat'];
    result_names{i,2} = [ROOT, 'data/ferrari/', DECONVNET, '/', cur_img_name, '.mat'];
    result_names{i,3} = [ROOT, 'data/ferrari/', HYBRID, '/', cur_img_name, '.mat'];
    result_names{i,4} = [ROOT, 'data/ferrari/', BASELINE, '/', cur_img_name, '.mat'];
    
    opts.imagePath = image_names{i};
    
    % Original image
    img = ksresize(imread(image_names{i}));
    viz_img = im2single(uint8(img));
    
    % Ground truth segmentation
    sz = size(img);
    seg = imresize(imread(seg_names{i}), [sz(1), sz(2)], 'nearest');
    viz_gt =  im2single(uint8(bsxfun(@times, single(seg), img)));
    
    % saliency
    res = load(result_names{i,1}); 
    viz_saliency = vl_imsc(bsxfun(@times, single(res.seg), img));
    %opts.algorithm = 'saliency';
    %[~, mask_saliency, ~] ...
    %        = hand_specified_neuron_viz_fn(opts);
    %mask_saliency = vl_imsc_am(mask_saliency);
    mask_saliency = vl_imsc_am(res.mask_signed);
    
    % Deconvnet
    res = load(result_names{i,2});
    viz_deconvnet = vl_imsc(bsxfun(@times, single(res.seg), img));
    %opts.algorithm = 'deconvnet';
    %[~, mask_deconvnet, ~] ...
    %        = hand_specified_neuron_viz_fn(opts);
    %mask_deconvnet = vl_imsc_am(mask_deconvnet);
    mask_deconvnet = vl_imsc_am(res.mask_signed);
    
    % Hybrid
    res = load(result_names{i,3});
    viz_am = vl_imsc(bsxfun(@times, single(res.seg), img));
    %opts.algorithm = 'TTT';
    %[~, mask_am, ~] ...
    %        = hand_specified_neuron_viz_fn(opts);
    %mask_am = vl_imsc_am(mask_am);
    mask_am = vl_imsc_am(res.mask_signed);
    
    % Baseline 2
    res = load(result_names{i,4});
    viz_baseline = vl_imsc(bsxfun(@times, single(res.seg), img));
    %mask_baseline = vl_imsc(baseline2(img));
    mask_baseline = repmat(vl_imsc(res.mask), [1,1,3]);
    
    viz{i} = [cat(2, pad_fn(viz_gt), pad_fn(viz_am), pad_fn(viz_saliency), pad_fn(viz_deconvnet), pad_fn(viz_baseline));
              cat(2, pad_fn(viz_img), pad_fn(mask_am), pad_fn(mask_saliency), pad_fn(mask_deconvnet), pad_fn(mask_baseline))];
end

viz_sizes = cell2mat(cellfun(@(x) size(x), viz, 'UniformOutput', false));
chosen_size = min(viz_sizes(:,2));

for i=1:NUM_IMAGES
    viz{i} = imresize(viz{i}, [NaN, chosen_size], 'nearest');
end

final_viz = cat(1, viz{:});

imwrite(final_viz, [FIGS_PATH, 'segmentation_101.png']);
