FIGS_PATH = 'supfigs/';

IMAGE_IDS = 10000 + [3:5:50];
IMAGE_NAMES = cell(numel(IMAGE_IDS),1);
for i=1:numel(IMAGE_IDS)
    IMAGE_NAMES{i} = sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(i));
end


MODEL_NAMES = {'../models/imagenet-vgg-verydeep-16.mat',...
               '../models/imagenet-caffe-alex.mat'};

%LAYERS = {[28, 42], [12, 20]};
%LAYERS = {[24, 36], [12, 20]};
LAYERS = {[5, 10, 17, 24, 31, 33, 35, 36], [4, 8, 10, 12, 15, 17, 19, 20]};

METHODS = {'deconvnet', 'TTT', 'saliency'};

viz_images = cell(numel(IMAGE_IDS), numel(MODEL_NAMES), numel(LAYERS{1}), ...
    numel(METHODS), 4);

for img_no = 1:numel(IMAGE_IDS)
    opts = struct();
    opts.randomizeWeights = false;
    opts.gpu = false;
    
    opts.relus_to_change = 1:100;
    opts.pools_to_change = 1:5;
    opts.convs_to_change = 1:100;
    
    opts.imagePath = IMAGE_NAMES{img_no};
    
    for model_no = 1:numel(MODEL_NAMES)
        opts.modelPath = MODEL_NAMES{model_no};
        
        for layer_no = 1:numel(LAYERS{model_no})
            opts.layer = LAYERS{model_no}(layer_no);
            
            for method_no = 1:numel(METHODS)
                
                cituation_no = 1;
                
                opts.neuron_I = 1/2;
                opts.neuron_J = 1/2;
                opts.neuron_channel = inf;
                
                opts.algorithm = METHODS{method_no};
                
                [~, viz_images{img_no, model_no, layer_no, method_no, cituation_no}, opts_new, ~] ...
                    = hand_specified_neuron_viz_fn(opts);
                
                viz_images{img_no, model_no, layer_no, method_no, cituation_no} = ...
                    padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, method_no, cituation_no}), [1, 1], 1, 'both');
                
                cituation_no = 2;
                
                opts.neuron_I = opts_new.neuron_I;
                opts.neuron_J = opts_new.neuron_J;
                opts.neuron_channel = -1;
                
                [~, viz_images{img_no, model_no, layer_no, method_no, cituation_no}, opts_new2, NET] ...
                    = hand_specified_neuron_viz_fn(opts);
                
                if(opts_new2.neuron_channel == opts_new.neuron_channel)
                    'Oops I randomly picked the maximally excited neuron'
                    keyboard;
                end
                
                viz_images{img_no, model_no, layer_no, method_no, cituation_no} = ...
                    padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, method_no, cituation_no}), [1,1], 1, 'both');
                
                cituation_no = 3;
                
                img = imread(IMAGE_NAMES{img_no});
                img_pp = opts_new.normalize(img);
                NET_info = vl_simplenn_display(NET, 'inputSize', [NET.meta.normalization.imageSize(1:2), 3, 1]);
                dzdy = zeros(NET_info.dataSize(:, end)', 'single');
                dzdy(opts_new.neuron_I, opts_new.neuron_J, :) = abs(randn([1,1,size(dzdy,3)], 'single'));
                %dzdy = abs(randn(NET_info.dataSize(:, end)', 'single'));
                res = vl_simplenn(NET, img_pp, dzdy, [], 'conserveMemory', true);
                viz_images{img_no, model_no, layer_no, method_no, cituation_no} = ...
                    padarray(vl_imsc_am(res(1).dzdx), [1,1], 1, 'both');

                clear NET NET_info dzdy res img img_pp opt_new opts_new2;
                
            end
        end
    end
end

orig_images = cell(numel(IMAGE_IDS), numel(MODEL_NAMES));

for model_no = 1:numel(MODEL_NAMES)
    NET = vl_simplenn_tidy(load(MODEL_NAMES{model_no}));
    for img_no = 1:numel(IMAGE_IDS)
        img = imread(IMAGE_NAMES{img_no});
        orig_images{img_no, model_no} = padarray(im2single(resizencrop(img, NET.meta.normalization.imageSize(1:2))), [1,1], 1, 'both');

        if(strfind(MODEL_NAMES{model_no}, 'vgg-verydeep-16'))
          model_str = 'vgg16';
        elseif(strfind(MODEL_NAMES{model_no}, 'alex'))
          model_str = 'alex';
        end
           
        imwrite(orig_images{img_no, model_no}, sprintf([FIGS_PATH, 'effect_of_phase_center_%s_%d.png'], model_str, img_no));

        clear img;
    end
    clear NET;
end

for method_no = 1:numel(METHODS)

final_image_vgg16 = [];
final_image_alexnet = [];

for img_no = 1:5%numel(IMAGE_IDS)

final_image_vgg16 = cat(2, final_image_vgg16, [cat(1, viz_images{img_no,1,1:8,method_no,1}),...
                                               cat(1, viz_images{img_no,1,1:8,method_no,2}),...
                                               cat(1, viz_images{img_no,1,1:8,method_no,3})]);
final_image_vgg16 = cat(2, final_image_vgg16, ones(size(final_image_vgg16,1),10,3,'single'));

final_image_alexnet = cat(2, final_image_alexnet, [cat(1, viz_images{img_no,2,1:8,method_no,1}),...
                                               cat(1, viz_images{img_no,2,1:8,method_no,2}),...
                                               cat(1, viz_images{img_no,2,1:8,method_no,3})]);
final_image_alexnet = cat(2, final_image_alexnet, ones(size(final_image_alexnet,1),10,3,'single'));


end

imwrite(final_image_alexnet, sprintf([FIGS_PATH, 'effect_of_phase_center_alexnet_%s_vert.png'], METHODS{method_no}));
imwrite(final_image_vgg16, sprintf([FIGS_PATH, 'effect_of_phase_center_vgg16_%s_vert.png'], METHODS{method_no}));

end
