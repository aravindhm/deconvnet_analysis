% generates fig. 4 of the paper

FIGS_PATH = 'genfigs';

IMAGE_IDS = [234,289];
IMAGE_NAMES = {sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(1)), ...
    sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(2))};

MODEL_NAMES = {'models/imagenet-vgg-verydeep-16.mat',...
    'models/imagenet-caffe-alex.mat'};

LAYERS = {[24, 36], [12, 20]};

METHODS = {'deconvnet'}; % other options are 'TTT', 'saliency' 

viz_images = cell(2,2,2,1,3);

for img_no = 1:2
    opts = struct();
    opts.randomizeWeights = false;
    opts.gpu = false;
    
    opts.relus_to_change = 1:100;
    opts.pools_to_change = 1:5;
    opts.convs_to_change = 1:100;
    
    opts.imagePath = IMAGE_NAMES{img_no};
    
    for model_no = 1:2
        opts.modelPath = MODEL_NAMES{model_no};
        
        for layer_no = 1:2
            opts.layer = LAYERS{model_no}(layer_no);
            
            for method_no = 1:1
                
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
                    error('Oops I randomly picked the maximally excited neuron');
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
                
            end
        end
    end
end

% final_image_vgg16 = [viz_images{1,1,1,1}, viz_images{1,1,2,1};
%     viz_images{1,1,1,2}, viz_images{1,1,2,2};
%     viz_images{1,1,1,3}, viz_images{1,1,2,3};
%     viz_images{2,1,1,1}, viz_images{2,1,2,1};
%     viz_images{2,1,1,2}, viz_images{2,1,2,2};
%     viz_images{2,1,1,3}, viz_images{2,1,2,3}];
% 
% final_image_alexnet = [viz_images{1,2,1,1}, viz_images{1,2,2,1};
%     viz_images{1,2,1,2}, viz_images{1,2,2,2};
%     viz_images{1,2,1,3}, viz_images{1,2,2,3};
%     viz_images{2,2,1,1}, viz_images{2,2,2,1};
%     viz_images{2,2,1,2}, viz_images{2,2,2,2};
%     viz_images{2,2,1,3}, viz_images{2,2,2,3}];
for t = 1:2
%final_image_vgg16_TTT = [viz_images{t,1,1,1,1}, viz_images{t,1,1,1,2}, viz_images{t,1,1,1,3};
%    viz_images{t,1,2,1,1}, viz_images{t,1,2,1,2}, viz_images{t,1,2,1,3}];
final_image_vgg16_TTT = [viz_images{t,1,1,1,1}, viz_images{t,1,2,1,1};
                         viz_images{t,1,1,1,2}, viz_images{t,1,2,1,2};
                         viz_images{t,1,1,1,3}, viz_images{t,1,2,1,2}];
    
% final_image_vgg16_TTT = [viz_images{1,1,1,2,1}, viz_images{1,1,1,2,2}, viz_images{1,1,1,2,3};
%     viz_images{1,1,2,2,1}, viz_images{1,1,2,2,2}, viz_images{1,1,2,2,3}];

% final_image_vgg16_saliency = [viz_images{1,1,1,3,1}, viz_images{1,1,1,3,2}, viz_images{1,1,1,3,3};
%     viz_images{1,1,2,3,1}, viz_images{1,1,2,3,2}, viz_images{1,1,2,3,3}];

%final_image_alexnet_TTT = [viz_images{t,2,1,1,1}, viz_images{t,2,1,1,2}, viz_images{t,2,1,1,3};
%    viz_images{t,2,2,1,1}, viz_images{t,2,2,1,2}, viz_images{t,2,2,1,3}];

final_image_alexnet_TTT = [viz_images{t,2,1,1,1}, viz_images{t,2,2,1,1};
                           viz_images{t,2,1,1,2}, viz_images{t,2,2,1,2};
                           viz_images{t,2,1,1,3}, viz_images{t,2,2,1,3}];

% final_image_alexnet_TTT = [viz_images{1,2,1,2,1}, viz_images{1,2,1,2,2}, viz_images{1,2,1,2,3};
%     viz_images{1,2,2,2,1}, viz_images{1,2,2,2,2}, viz_images{1,2,2,2,3}];

% final_image_alexnet_saliency = [viz_images{1,2,1,3,1}, viz_images{1,2,1,3,2}, viz_images{1,2,1,3,3};
%     viz_images{1,2,2,3,1}, viz_images{1,2,2,3,2}, viz_images{1,2,2,3,3}];

% imwrite(final_image_alexnet_TTT, '../mahendran16cvpr/figs/fig2_alexnet_saliency_img234.png');
% imwrite(final_image_vgg16_TTT, '../mahendran16cvpr/figs/fig2_vgg16_saliency_img234.png');

imwrite(final_image_alexnet_TTT, sprintf([FIGS_PATH, '/neuronselectivity_alexnet_%s_img%d_vert.png'], METHODS{1}, IMAGE_IDS(t)));
imwrite(final_image_vgg16_TTT, sprintf([FIGS_PATH, '/neuronselectivity_vgg16_%s_img%d_vert.png'], METHODS{1}, IMAGE_IDS(t)));
end