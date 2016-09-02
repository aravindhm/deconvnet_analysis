% Generates the figure in supplementary material that shows that LRN^{BP}
% vs identity operations in the reversed architecture are not very
% different

FIGS_PATH = 'genfigs/';

% The effect of hacking the local contrast normalization layers

% we are using 1 image only and comparing our approach with local response
% normalization and without local response normalization. The latter is
% closer to deconvnet the former is closer to saliency.

% As VGG-19 doesn't have LRN layers, this figure only involves alexnet.

% We'll do it for one shallow and one deep layer

%IMAGE_IDS = [43];
IMAGE_NAMES = {'stock_abstract.jpg'};%, ...
   %sprintf([IMAGENET12_VAL_PATH, '/ILSVRC2012_val_%08d.JPEG'], IMAGE_IDS(1)) };

MODEL_NAMES = {'models/imagenet-caffe-alex.mat'};

LAYERS = {[15,20]}; % pool5 and relu7

METHOD_NAMES = {'TTT', 'TTTF'};

viz_images = cell(2,1,2,2);

for img_no = 1:1
    opts = struct();
    opts.randomizeWeights = false;
    opts.gpu = false;
    
    opts.relus_to_change = 1:100;
    opts.pools_to_change = 1:5;
    opts.convs_to_change = 1:100;
    
    opts.imagePath = IMAGE_NAMES{img_no};
    
    for model_no = 1
        opts.modelPath = MODEL_NAMES{model_no};
        
        for layer_no = 1:2
            opts.layer = LAYERS{model_no}(layer_no);
            
            for method_no = 1:2
                opts.algorithm = METHOD_NAMES{method_no};
            
                opts.neuron_I = inf;
                opts.neuron_J = inf;
                opts.neuron_channel = inf;
            
                [~, t, ~, ~] = hand_specified_neuron_viz_fn(opts);
            
                viz_images{img_no, model_no, layer_no, method_no} = ...
                    padarray(vl_imsc_am(t), [1, 1], 1, 'both');
            
            end
        end
    end
end

NET = vl_simplenn_tidy(load(MODEL_NAMES{1}));
img = im2single(imread(IMAGE_NAMES{1}));
img_pp = padarray(resizencrop(img, NET.meta.normalization.imageSize), [1,1], 1, 'both');

viz_final = [img_pp, viz_images{1,1,1,1}, viz_images{1,1,1,2}, viz_images{1,1,2,1}, viz_images{1,1,2,2}];
%viz_images{2,1,1,1}, viz_images{2,1,1,2}, viz_images{2,1,2,1}, viz_images{2,1,2,2}];

imwrite(viz_final, [FIGS_PATH, 'effectof_lrnnobackprop.jpg']);   