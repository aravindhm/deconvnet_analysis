FIGS_PATH = 'supfigs/';

IMAGE_IDS = 10000 + [4:5:50];
IMAGE_NAMES = cell(numel(IMAGE_IDS),1);
for i=1:numel(IMAGE_IDS)
    IMAGE_NAMES{i} = sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(i));
end

MODEL_NAMES = {'../models/imagenet-vgg-verydeep-16.mat',...
               '../models/imagenet-caffe-alex.mat'};

%LAYERS = {[28, 42], [12, 20]};
LAYERS = {[5, 10, 17, 24, 31, 33, 35, 36], [4, 8, 10, 12, 15, 17, 19, 20]};
%LAYERS = {[31, 36], [15, 20]};

METHODS = {'deconvnet', 'saliency', 'TTT'};

viz_images = cell(5,2,8,4);

for img_no = 1:numel(IMAGE_IDS)
    opts = struct();
    
    opts.gpu = false;

    opts.relus_to_change = 1:100;
    opts.pools_to_change = 1:5;
    opts.convs_to_change = 1:100;

    opts.neuron_I = 1/2;
    opts.neuron_J = 1/2;
    opts.neuron_channel = inf;
    
    opts.imagePath = IMAGE_NAMES{img_no};
    
    for model_no = 1:2
        opts.modelPath = MODEL_NAMES{model_no};
        
        for layer_no = 1:numel(LAYERS{model_no})
            opts.layer = LAYERS{model_no}(layer_no);
            
            for method_no = 1:numel(METHODS)
                opts.randomizeWeights = false;
                opts.algorithm = METHODS{method_no};
                
                [~, viz_images{img_no, model_no, layer_no, method_no}, ~] ...
                    = hand_specified_neuron_viz_fn(opts);
                viz_images{img_no, model_no, layer_no, method_no} = ...
                    padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, method_no}), [1,1], 1, 'both');
                
                
            end
            
            %opts.algorithm = 'FFF';
            %opts.randomizeWeights = true;
            %
            %[~, viz_images{img_no, model_no, layer_no, 5}, ~] ...
            %        = hand_specified_neuron_viz_fn(opts);
            %viz_images{img_no, model_no, layer_no, 5} = ...
            %        padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, 5}), [1,1], 1, 'both');
            
        end
        
        img = imread(IMAGE_NAMES{img_no});
        NET = vl_simplenn_tidy(load(opts.modelPath));
        viz_images{img_no, model_no, 1, 4} = padarray(im2single(resizencrop(img, NET.meta.normalization.imageSize(1:2))), [1,1], 1, 'both');
        clear NET;
        clear img;
        
    end

end

final_image_vgg16 = ...
[cat(1, viz_images{:,1,1,4}), cat(1, viz_images{:,1,1,1}), cat(1, viz_images{:,1,1,2}), cat(1, viz_images{:,1,1,3}),...
 cat(1, viz_images{:,1,2,1}), cat(1, viz_images{:,1,2,2}), cat(1, viz_images{:,1,2,3}),...
 cat(1, viz_images{:,1,3,1}), cat(1, viz_images{:,1,3,2}), cat(1, viz_images{:,1,3,3}),...
 cat(1, viz_images{:,1,4,1}), cat(1, viz_images{:,1,4,2}), cat(1, viz_images{:,1,4,3}),...
 cat(1, viz_images{:,1,5,1}), cat(1, viz_images{:,1,5,2}), cat(1, viz_images{:,1,5,3}),...
 cat(1, viz_images{:,1,6,1}), cat(1, viz_images{:,1,6,2}), cat(1, viz_images{:,1,6,3}),...
 cat(1, viz_images{:,1,7,1}), cat(1, viz_images{:,1,7,2}), cat(1, viz_images{:,1,7,3}),...
 cat(1, viz_images{:,1,8,1}), cat(1, viz_images{:,1,8,2}), cat(1, viz_images{:,1,8,3})];

final_image_alexnet = ...
[cat(1, viz_images{:,2,1,4}), cat(1, viz_images{:,2,1,1}), cat(1, viz_images{:,2,1,2}), cat(1, viz_images{:,2,1,3}),...
 cat(1, viz_images{:,2,2,1}), cat(1, viz_images{:,2,2,2}), cat(1, viz_images{:,2,2,3}),...
 cat(1, viz_images{:,2,3,1}), cat(1, viz_images{:,2,3,2}), cat(1, viz_images{:,2,3,3}),...
 cat(1, viz_images{:,2,4,1}), cat(1, viz_images{:,2,4,2}), cat(1, viz_images{:,2,4,3}),...
 cat(1, viz_images{:,2,5,1}), cat(1, viz_images{:,2,5,2}), cat(1, viz_images{:,2,5,3}),...
 cat(1, viz_images{:,2,6,1}), cat(1, viz_images{:,2,6,2}), cat(1, viz_images{:,2,6,3}),...
 cat(1, viz_images{:,2,7,1}), cat(1, viz_images{:,2,7,2}), cat(1, viz_images{:,2,7,3}),...
 cat(1, viz_images{:,2,8,1}), cat(1, viz_images{:,2,8,2}), cat(1, viz_images{:,2,8,3}) ];

imwrite(final_image_vgg16, [FIGS_PATH, 'fig_spatial_selectivity_noarcheffects_vgg16.png']);
imwrite(final_image_alexnet, [FIGS_PATH, 'fig_spatial_selectivity_noarcheffects_alexnet.png']);
