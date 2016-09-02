% generates fig 6 in the paper

FIGS_PATH = 'genfigs/';

IMAGE_IDS = [2, 56, 94, 112, 131];

IMAGE_NAMES = {sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(1)), ...
     sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(2)),...
     sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(3)),...
     sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(4)),...
     sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(5))...
     };

MODEL_NAMES = {'models/imagenet-vgg-verydeep-16.mat',...
    'models/imagenet-caffe-alex.mat'};

LAYERS = {[31, 36], [15, 20]};

METHODS = {'deconvnet', 'saliency', 'TTT'};

viz_images = cell(5,2,2,4);

for img_no = 1:5
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
        
        for layer_no = 1:2
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
 cat(1, viz_images{:,1,2,1}), cat(1, viz_images{:,1,2,2}), cat(1, viz_images{:,1,2,3})];

final_image_alexnet = ...
[cat(1, viz_images{:,2,1,4}), cat(1, viz_images{:,2,1,1}), cat(1, viz_images{:,2,1,2}), cat(1, viz_images{:,2,1,3}),...
 cat(1, viz_images{:,2,2,1}), cat(1, viz_images{:,2,2,2}), cat(1, viz_images{:,2,2,3})];

imwrite(final_image_vgg16, [FIGS_PATH, 'fig_spatial_selectivity_noarcheffects_vgg16.jpg']);
imwrite(final_image_alexnet, [FIGS_PATH, 'fig_spatial_selectivity_noarcheffects_alexnet.jpg']);

%final_image_vgg16_img170 = [viz_images{1,1,1,1}, viz_images{1,1,1,2}, viz_images{1,1,1,3}, viz_images{1,1,1,4}, viz_images{1,1,1,5};
%               viz_images{1,1,2,1}, viz_images{1,1,2,2}, viz_images{1,1,2,3}, viz_images{1,1,2,4}, viz_images{1,1,2,5}];
           
%final_image_vgg16_img98 = [viz_images{2,1,1,1}, viz_images{2,1,1,2}, viz_images{2,1,1,3}, viz_images{2,1,1,4}, viz_images{2,1,1,5};
%               viz_images{2,1,2,1}, viz_images{2,1,2,2}, viz_images{2,1,2,3}, viz_images{2,1,2,4}, viz_images{2,1,2,5}];

%final_image_alexnet_img170 = [viz_images{1,2,1,1}, viz_images{1,2,1,2}, viz_images{1,2,1,3}, viz_images{1,2,1,4}, viz_images{1,2,1,5};
%               viz_images{1,2,2,1}, viz_images{1,2,2,2}, viz_images{1,2,2,3}, viz_images{1,2,2,4}, viz_images{1,2,2,5}];
           
%final_image_alexnet_img98 = [viz_images{2,2,1,1}, viz_images{2,2,1,2}, viz_images{2,2,1,3}, viz_images{2,2,1,4}, viz_images{2,2,1,5};
%               viz_images{2,2,2,1}, viz_images{2,2,2,2}, viz_images{2,2,2,3}, viz_images{2,2,2,4}, viz_images{2,2,2,5}];


%imwrite(final_image_alexnet_img170, '../mahendran16cvpr/figs/fig_spatial_selectivity_alexnet_img170.jpg');
%imwrite(final_image_alexnet_img98, '../mahendran16cvpr/figs/fig_spatial_selectivity_alexnet_img98.jpg');

%imwrite(final_image_vgg16_img170, '../mahendran16cvpr/figs/fig_spatial_selectivity_vgg16_img170.jpg');
%imwrite(final_image_vgg16_img98, '../mahendran16cvpr/figs/fig_spatial_selectivity_vgg16_img98.jpg');