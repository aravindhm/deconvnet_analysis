% Generates fig 1 in the paper
FIGS_PATH = 'genfigs/';

IMAGE_IDS = [170, 98, 234,289,299,27,77];
IMAGE_NAMES = cell(numel(IMAGE_IDS),1);
for i=1:numel(IMAGE_IDS)
    IMAGE_NAMES{i} = sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(i));
end
    

MODEL_NAMES = {'models/imagenet-vgg-verydeep-16.mat',...
    'models/imagenet-caffe-alex.mat'};

%LAYERS = {[28, 42], [12, 20]};
LAYERS = {[36], [20]};

METHODS = {'deconvnet', 'saliency', 'TTT'};

viz_images = cell(numel(IMAGE_IDS),2,1,4);

for img_no = 1:numel(IMAGE_IDS)
    opts = struct();
    opts.randomizeWeights = false;
    opts.gpu = false;

    opts.relus_to_change = 1:100;
    opts.pools_to_change = 1:5;
    opts.convs_to_change = 1:100;

    opts.neuron_I = inf;
    opts.neuron_J = inf;
    opts.neuron_channel = inf;
    
    opts.imagePath = IMAGE_NAMES{img_no};
    
    for model_no = 1:2
        opts.modelPath = MODEL_NAMES{model_no};
        
        for layer_no = 1:1
            opts.layer = LAYERS{model_no}(layer_no);
            
            for method_no = 1:numel(METHODS)
                opts.algorithm = METHODS{method_no};
                
                [~, viz_images{img_no, model_no, layer_no, method_no}, opts_new, ~] ...
                    = hand_specified_neuron_viz_fn(opts);
                viz_images{img_no, model_no, layer_no, method_no} = ...
                    padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, method_no}), [1,1], 1, 'both');
                
                
            end
        end
        img = imread(IMAGE_NAMES{img_no});
        NET = vl_simplenn_tidy(load(opts.modelPath));
        viz_images{img_no, model_no, 1, 4} = padarray(im2single(resizencrop(img, NET.meta.normalization.imageSize(1:2))), [1,1], 1, 'both');    
        clear NET;
        clear img;
    end
    
    
end

t = 1;
final_image_vgg16 = [...
               cat(2, viz_images{:,t,1,4});...
               cat(2, viz_images{:,t,1,1});...
               cat(2, viz_images{:,t,1,2});...
               cat(2, viz_images{:,t,1,3})];
           
t = 2;
final_image_alexnet = [...
               cat(2, viz_images{:,t,1,4});...
               cat(2, viz_images{:,t,1,1});...
               cat(2, viz_images{:,t,1,2});...
               cat(2, viz_images{:,t,1,3})];
           
imwrite(final_image_alexnet, [FIGS_PATH, 'fig_splash_alexnet.jpg']);
imwrite(final_image_vgg16, [FIGS_PATH, 'fig_splash_vgg16.jpg']);
