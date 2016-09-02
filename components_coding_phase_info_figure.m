% generates fig. 3 of the paper

FIGS_PATH = 'genfigs/';

%IMAGE_IDS = [77, 170];%, 98, 234,289,295,298,299,306];
IMAGE_IDS = [299];
IMAGE_NAMES = {sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(1))};

MODEL_NAMES = {'models/imagenet-vgg-verydeep-16.mat'};

LAYERS = {[31]};

% Relu mask (T/F), Pooling Switches (T/F), Relu Backwards (T/F)
METHODS = {'TTT', 'TFT', 'FFT', 'FTT', 'TTF', 'TFF', 'FFF', 'FTF'};

viz_images = cell(1,1,1,8);

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
        
        for layer_no = 1
            opts.layer = LAYERS{model_no}(layer_no);
            
            for method_no = 1:numel(METHODS)
                opts.algorithm = METHODS{method_no};
                
                opts.neuron_I = inf;
                opts.neuron_J = inf;
                opts.neuron_channel = inf; 
                
                [~, viz_images{img_no, model_no, layer_no, method_no}, ~, ~] ...
                     = hand_specified_neuron_viz_fn(opts);
                 
                viz_images{img_no, model_no, layer_no, method_no} = ...
                     padarray(vl_imsc_am(viz_images{img_no, model_no, layer_no, method_no}), [1, 1], 1, 'both');
            end
            
        end
    end
end

final_image_relubackwards = [viz_images{1,1,1,1}, viz_images{1,1,1,4};
               viz_images{1,1,1,2}, viz_images{1,1,1,3}];
           
final_image_norelubackwards = [viz_images{1,1,1,5}, viz_images{1,1,1,6};
               viz_images{1,1,1,8}, viz_images{1,1,1,7}];

imwrite(final_image_relubackwards, [FIGS_PATH, 'phasesources_relubackwards.jpg']);
imwrite(final_image_norelubackwards, [FIGS_PATH, 'phasesources_norelubackwards.jpg']);

imwrite(viz_images{1,1,1,1}, [FIGS_PATH, 'relubackward_relumask_mpoolswitches.jpg']);
imwrite(viz_images{1,1,1,2}, [FIGS_PATH, 'relubackward_relumask_mpoolcenter.jpg']);
imwrite(viz_images{1,1,1,3}, [FIGS_PATH, 'relubackward_norelumask_mpoolcenter.jpg']);
imwrite(viz_images{1,1,1,4}, [FIGS_PATH, 'relubackward_norelumask_mpoolswitches.jpg']);

imwrite(viz_images{1,1,1,5}, [FIGS_PATH, 'norelubackward_relumask_mpoolswitches.jpg']);
imwrite(viz_images{1,1,1,6}, [FIGS_PATH, 'norelubackward_relumask_mpoolcenter.jpg']);
imwrite(viz_images{1,1,1,7}, [FIGS_PATH, 'norelubackward_norelumask_mpoolcenter.jpg']);
imwrite(viz_images{1,1,1,8}, [FIGS_PATH, 'norelubackward_norelumask_mpoolswitches.jpg']);
