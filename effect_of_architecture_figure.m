% Generates fig. 5 of the paper

FIGS_PATH = 'genfigs/';

%IMAGE_IDS = [77, 170];%, 98, 234,289,295,298,299,306];
IMAGE_IDS = [77];
IMAGE_NAMES = {sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(1))};%, ...

MODEL_NAMES = {'models/imagenet-vgg-verydeep-16.mat',...
    'models/imagenet-caffe-alex.mat'};

%LAYERS = {[37,39], [15,17]};
LAYERS = {[31,36], [15,20]};
viz_images = cell(1,1,2,7);

for img_no = 1
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
            opts.randomizeWeights = false;
            opts.gpu = false;
            
            opts.layer = LAYERS{model_no}(layer_no);
            
            % The the deconvnet
            opts.algorithm = 'deconvnet';
            
            opts.neuron_I = 1/2;
            opts.neuron_J = 1/2;
            opts.neuron_channel = inf;
            
            [~, t, ~, ~] = hand_specified_neuron_viz_fn(opts);
            
            viz_images{img_no, model_no, layer_no, 1} = ...
                padarray(vl_imsc_am(t), [1, 1], 1, 'both');
            
            % But all the pixels are active
            viz_images{img_no, model_no, layer_no, 2} = ...
                padarray(vl_imsc_am(sign(t).*abs(t).^0.1), [1,1], 1, 'both');
            
            % First our algorithm
            opts.algorithm = 'TTT';
            
            opts.neuron_I = 1/2;
            opts.neuron_J = 1/2;
            opts.neuron_channel = inf;
            
            [~, t, ~, ~] = hand_specified_neuron_viz_fn(opts);
            
            viz_images{img_no, model_no, layer_no, 3} = ...
                padarray(vl_imsc_am(t), [1, 1], 1, 'both');
            
            % But all the pixels are active
            viz_images{img_no, model_no, layer_no, 4} = ...
                padarray(vl_imsc_am(sign(t).*abs(t).^0.1), [1,1], 1, 'both');
            
            % What if we remove everything - we still see the foveation
            opts.algorithm = 'FFF';
            [~, viz_images{img_no, model_no, layer_no, 5}, opts_new, ~] ...
                = hand_specified_neuron_viz_fn(opts);
            
            viz_images{img_no, model_no, layer_no, 5} = ...
                padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, 5}), [1, 1], 1, 'both');
            
            % This has nothing to do with the weights. See it is there even
            % if we randomize the weights.
            opts.randomizeWeights = true;
            opts.algorithm = 'FFF';
            opts.neuron_I = opts_new.neuron_I; % After randomizing the weights the max response
            % is going to be at a different location. So this is to ensure
            % that we use the same neuron as above.
            opts.neuron_J = opts_new.neuron_J;
            opts.neuron_channel = opts_new.neuron_channel;
            [~, viz_images{img_no, model_no, layer_no, 6}, ~, ~] ...
                = hand_specified_neuron_viz_fn(opts);
            
            viz_images{img_no, model_no, layer_no, 6} = ...
                padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, 6}), [1, 1], 1, 'both');

            % And it's like this for AlexNet.
            opts_alex = opts;
            opts_alex.modelPath = MODEL_NAMES{2};
            opts_alex.randomizeWeights = true;
            opts_alex.algorithm = 'FFF';
            opts_alex.neuron_I = inf;
            opts_alex.neuron_J = inf;
            opts_alex.neuron_channel = inf;
            opts_alex.layer = LAYERS{2}(layer_no);
            
            [~, viz_images{img_no, model_no, layer_no, 7}, ~, ~] ...
                = hand_specified_neuron_viz_fn(opts_alex);
            
            viz_images{img_no, model_no, layer_no, 7} = ...
                padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, 7}), [1, 1], 1, 'both');
            
            
        end
    end
end

img = imread(IMAGE_NAMES{1});
img_pp = resizencrop(img, size(viz_images{1,1,1,1}));
final_image = [cat(2, viz_images{1,1,1,[3,4,5,6]})];%...
    %cat(2, viz_images{1,1,2,1:6})];

%final_image_alexnet = [viz_images{1,1,1,7}, viz_images{1,1,2,7}];

%figure; subplot(1,2,1); imshow(final_image); subplot(1,2,2); imshow(final_image_alexnet);
imwrite(final_image, [FIGS_PATH, 'architectureeffect_concise.jpg']);
%imwrite(final_image_alexnet, [FIGS_PATH, 'fig6_architectureeffect_alexnet.jpg']);