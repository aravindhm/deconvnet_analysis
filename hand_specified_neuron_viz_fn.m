function [viz, template, opts, NET, img] = hand_specified_neuron_viz_fn(opts)
% This is the main engine - it reads many different options and creates a
% reversed network and visualizes some neuron with that reversed network
% 
% opts can be like below :-
%
% randomizeWeights: 0 - whether or not to randomize network weights
%
% gpu: 0 - boolean - whether or not to use the gpu
%
% relus_to_change: [1x100 double] - Which relu's in the network do you want
% to change when creating the reversed architecture. if set to [1] it will
% change the first relu in the network, [1,2] will change the first 2.
% Similarly, [1,3] will change the first and the third relu but skip the
% second one. If left unchanged they do BP in the reversed architecture
%
% pools_to_change: [1 2 3 4 5] - same as above but for pooling layers
%
% convs_to_change: [1x100 double] - same as above but for convs. 
%
% neuron_I: Inf - This is the spatial location of which neuron to visualize
% Inf for the maximally active neuron
% -1 for the last row of neurons
% 1/2 or some fraction between 0 and 1 to pick a relative location in the
% spatial field of view
% or an integer to directly specify the neuron location/
%
% neuron_J: Inf 
%
% neuron_channel: Inf
%
% Please be very careful when using the special cases (inf, fraction, -1
% etc) ... read the code and debug to check which neuron got picked.
%
% imagePath: 'imagenet12-val/ILSVRC2012_val_00000170.JPEG' - which image to
% visualize over.
%
% modelPath: 'models/imagenet-vgg-verydeep-16.mat' - which forward network
% to visualize
% 
% layer: 36 - which layer in the forward network to visualize
%
% algorithm: 'deconvnet' - which algorithm to use. There are many many
% options. Please have a look at the switch case below. 
% Prominently - 'deconvnet', 'saliency', 'TTT'(for DeSaliNet) can be used.
%
% Opts can also be a more complete version like
% randomizeWeights: 0
% gpu: 0
% relus_to_change: [1x100 double]
% pools_to_change: [1 2 3 4 5]
% convs_to_change: [1x100 double]
% neuron_I: 1
% neuron_J: 1
% neuron_channel: 159
% imagePath: 'imagenet12-val/ILSVRC2012_val_00000170.JPEG'
% modelPath: 'models/imagenet-vgg-verydeep-16.mat'
% layer: 36
% algorithm: 'deconvnet'
% use_relu_mask: 0
% use_pooling_switches: 1
% relu_backward: 1
% lrn_nobackprop: 1
% conv_exciteonly: 0
% normalize: [function_handle]
% denormalize: @(x)bsxfun(@plus,x,NET.meta.normalization.averageImage)
%
% Again, since there are many options it is best to run an image and step
% through the code to see what is happening to make sure it is constructing
% the right network.
%
% Author: Aravindh Mahendran (Copyright 2016-17)
% University of Oxford

if(~isfield(opts, 'gpu'))
    opts.gpu = false;
end

%% Settings based on the chosen algorithm
switch opts.algorithm
    case {'deconvnet_noisy'}
        opts.use_noisy_relu = true;
        opts.lrn_nobackprop = true;
    case {'deconvnet', 'FTT'}
        opts.use_relu_mask = false;
        opts.use_pooling_switches = true;
        opts.relu_backward = true;
        opts.lrn_nobackprop = true;        
    case {'hybrid', 'TTT'}
        opts.use_relu_mask = true;
        opts.use_pooling_switches = true;
        opts.relu_backward = true;
        opts.lrn_nobackprop = true;
    case 'TTTF'
        opts.use_relu_mask = true;
        opts.use_pooling_switches = true;
        opts.relu_backward = true;
        opts.lrn_nobackprop = false;
    case 'TTTFT'
        opts.use_relu_mask = true;
        opts.use_pooling_switches = true;
        opts.relu_backward = true;
        opts.lrn_nobackprop = false;
        opts.conv_exciteonly = true;
    case {'saliency'}
        opts.use_relu_mask = true;
        opts.use_pooling_switches = true;
        opts.relu_backward = false;
        opts.lrn_nobackprop = false;
    case {'TTF'}
        opts.use_relu_mask = true;
        opts.use_pooling_switches = true;
        opts.relu_backward = false;
        opts.lrn_nobackprop = true; % This is the only bit that is different from 'Saliency'
    case {'TTFF'}
        opts.use_relu_mask = true;
        opts.use_pooling_switches = true;
        opts.relu_backward = false;
        opts.lrn_nobackprop = false; % we want to do the conservative and non conservative backprop together
    case {'deconvnet_unpooltocenter', 'FFT'}
        opts.use_relu_mask = false;
        opts.use_pooling_switches = false;
        opts.relu_backward = true;
        opts.lrn_nobackprop = true;
    case 'TFT'
        opts.use_relu_mask = true;
        opts.use_pooling_switches = false;
        opts.relu_backward = true;
        opts.lrn_nobackprop = true;
    case 'TFF'
        opts.use_relu_mask = true;
        opts.use_pooling_switches = false;
        opts.relu_backward = false;
        opts.lrn_nobackprop = true;
    case 'FFF'
        opts.use_relu_mask = false;
        opts.use_pooling_switches = false;
        opts.relu_backward = false;
        opts.lrn_nobackprop = true;
    case 'FTF'
        opts.use_relu_mask = false;
        opts.use_pooling_switches = true;
        opts.relu_backward = false;
        opts.lrn_nobackprop = true;
    otherwise
        error('Unknown algorithm %s\n', opts.algorithm);
end

if(strcmp(opts.algorithm, 'TTTFT'))
    opts.conv_exciteonly = true;
else
    opts.conv_exciteonly = false;
end

%% Load the network and prune down the layers

NET = load(opts.modelPath);
NET = vl_simplenn_tidy(NET);
NET.layers = NET.layers(1:opts.layer); 

%% Randomize the layer weights
if(opts.randomizeWeights)
    for i=1:numel(NET.layers)
        if(strcmp(NET.layers{i}.type, 'conv'))
            type = 'single';
            sz = size(NET.layers{i}.weights{1});
            h = sz(1); w = sz(2); in = sz(3); out = sz(4);
            %sc = sqrt(2/(h*w*out)) ;
            %NET.layers{i}.weights{1} = randn(h, w, in, out, type)*sc ;
            %NET.layers{i}.weights{2} = zeros(out, 1, type);
            sc = 0.01;
            NET.layers{i}.weights{1} = randn(h, w, in, out, type)*sc;
            NET.layers{i}.weights{2} = zeros(out, 1, type);
        end
    end
end

%% Create a network without the local response normalization layers. 
% This is used to find receptive field sizes for the hybrid and deconvnet method

NET_nolrn = NET;
relu_layer.type = 'relu';
relu_layer.leak = 0;
for i=1:numel(NET_nolrn.layers)
  if strcmp(NET_nolrn.layers{i}.type, 'normalize')
    NET_nolrn.layers{i} = relu_layer;
  end
  if strcmp(NET_nolrn.layers{i}.type, 'lrn')
    NET_nolrn.layers{i} = relu_layer;
  end
  if strcmp(NET_nolrn.layers{i}.type, 'conv')
    NET_nolrn.layers{i}.weights{1} = ones(size(NET.layers{i}.weights{1}), 'single');
    NET_nolrn.layers{i}.weights{2} = ones(size(NET.layers{i}.weights{2}), 'single');
  end
end
NET_nolrn = vl_simplenn_tidy(NET_nolrn);

%% Collect network information
% This is also mostly for receptive field measurement

NET_info = vl_simplenn_display(NET, 'inputSize', [NET.meta.normalization.imageSize(1:3), 1]);
NET_nolrn_info = vl_simplenn_display(NET_nolrn, 'inputSize', [NET.meta.normalization.imageSize(1:3), 1]);
NUM_CHANNELS = NET_info.dataSize(3, end);

%% Change the RELUs based on the opts

if(~isempty(opts.relus_to_change))

    
counter = 0;
for i=1:numel(NET.layers)
    if (strcmp(NET.layers{i}.type, 'relu'))
        counter = counter + 1;
        if(find(opts.relus_to_change == counter))
            if(isfield('opts', 'use_noisy_relu') && opts.use_noisy_relu)
                NET.layers{i}.type = 'relu_noisy';
            elseif(opts.use_relu_mask && opts.relu_backward)
                NET.layers{i}.type = 'relu_eccv16';
            elseif (~opts.use_relu_mask && opts.relu_backward)
                NET.layers{i}.type = 'relu_deconvnet';
            elseif (~opts.use_relu_mask && ~opts.relu_backward)
                NET.layers{i}.type = 'relu_nobackprop';
            end
        end
    end
end

end


if(opts.conv_exciteonly && ~isempty(opts.convs_to_change))

    
counter = 0;
for i=1:numel(NET.layers)
    if (strcmp(NET.layers{i}.type, 'conv'))
        counter = counter + 1;
        if(find(opts.convs_to_change == counter))
           NET.layers{i}.type = 'conv_exciteonly';
        end
    end
end

end

if(~opts.use_pooling_switches && ~isempty(opts.pools_to_change))

counter = 0;
for i=1:numel(NET.layers)
    if (strcmp(NET.layers{i}.type, 'pool'))
        counter = counter + 1;
        if(find(opts.pools_to_change == counter))
           NET.layers{i}.type = 'pool_center';
        end
    end
end

end

if(opts.lrn_nobackprop)
    for i=1:numel(NET.layers)
        if strcmp(NET.layers{i}.type, 'normalize')
            NET.layers{i}.type = 'normalize_nobackprop';
        end
        if strcmp(NET.layers{i}.type, 'lrn')
            NET.layers{i}.type = 'lrn_nobackprop';
        end
    end
end

%% Move NET to GPU
if(opts.gpu)
    NET_GPU = vl_simplenn_move(NET, 'gpu');
else
    NET_GPU = NET;
end

%% Setup the variables and functions for operating the network

opts.normalize = @(x) bsxfun(@minus, single(resizencrop(x, NET.meta.normalization.imageSize(1:2))), NET.meta.normalization.averageImage);
opts.denormalize = @(x) bsxfun(@plus, x, NET.meta.normalization.averageImage);

%% Run over the neurons and generate the visuals.

% read the image and evaluate the network on it
img = imread(opts.imagePath);
if(opts.gpu)
    img_pp = gpuArray(opts.normalize(img)); %, NET.meta.normalization.averageImage);
else
    img_pp = opts.normalize(img);
end

sz = NET_info.dataSize(:, end)';

% set the top derivative to be 1 at the position of maximal response
if(opts.gpu)
    dzdy = zeros(sz, 'single', 'gpuArray');
else
    dzdy = zeros(sz, 'single');
end

% Pick the neuron baesd on the parameters in opts

if(opts.neuron_I < 1 && opts.neuron_I ~= -1)
  opts.neuron_I = max( round(size(dzdy, 1) * opts.neuron_I), 1);
elseif(opts.neuron_I == -1)
  opts.neuron_I = size(dzdy, 1);
end

if(opts.neuron_J < 1 && opts.neuron_J ~= -1)
  opts.neuron_J = max(round(size(dzdy, 2) * opts.neuron_J), 1);
elseif(opts.neuron_J == -1)
  opts.neuron_J = size(dzdy, 2);
end

if(isinf(opts.neuron_I)) % find the maximally firing neuron
   res = vl_simplenn(NET_GPU, img_pp);
   if(isinf(opts.neuron_channel))
     [~, argmax] = max(res(end).x(:));
     [opts.neuron_I, opts.neuron_J, opts.neuron_channel] = ...
        ind2sub(size(res(end).x), argmax);
   else
     [~, argmax] = max( ...
         reshape( res(end).x(:, :, opts.neuron_channel), [], 1) );
     [opts.neuron_I, opts.neuron_J, ~] = ind2sub(size(res(end).x), argmax);
   end
else
    res = vl_simplenn(NET_GPU, img_pp);
    if(isinf(opts.neuron_channel))
        [~, opts.neuron_channel] = max( ...
            res(end).x(opts.neuron_I, opts.neuron_J, :), [], 3);
    end
end


if (opts.neuron_channel ~= -1)

  dzdy(opts.neuron_I, opts.neuron_J, opts.neuron_channel) = 1;
  res2 = vl_simplenn(NET_GPU, img_pp, dzdy);
  template = gather(res2(1).dzdx);

else
  while(true)
      neuron_channel = randi(sz(3), 1);
      dzdy(opts.neuron_I, opts.neuron_J, neuron_channel) = 1;
      res2 = vl_simplenn(NET_GPU, img_pp, dzdy);
      template = gather(res2(1).dzdx);
      pos = find(template ~= 0, 1);
      if(isempty(pos))
          dzdy(opts.neuron_I, opts.neuron_J, neuron_channel) = 0;
      else
          break;
      end
  end
  opts.neuron_channel = neuron_channel;
end

%rf_start_pos = [opts.neuron_I - 1; opts.neuron_J - 1] .* ...
%    NET_nolrn_info.receptiveFieldStride(:, end) + ...
%    NET_nolrn_info.receptiveFieldOffset(:, end) - ...
%    ceil(NET_nolrn_info.receptiveFieldSize(:, end) / 2) + 1;
%rf_start_pos = max(rf_start_pos, 1);
%
%rf_end_pos = min(rf_start_pos + NET_nolrn_info.receptiveFieldSize(:, end) - 1, ...
%    NET.meta.normalization.imageSize(1:2)') ;
%
%viz = vl_imsc(template(rf_start_pos(1):rf_end_pos(1), ...
%                       rf_start_pos(2):rf_end_pos(2), :));

img = resizencrop(img, NET.meta.normalization.imageSize(1:2));

rf_start_pos = [opts.neuron_I - 1; opts.neuron_J - 1] .* ...
    NET_nolrn_info.receptiveFieldStride(:, end) + ...
    NET_nolrn_info.receptiveFieldOffset(:, end) - ...
    ceil(NET_nolrn_info.receptiveFieldSize(:, end) / 2) + 1;

rf_start_pos =ceil(rf_start_pos);

rf_end_pos = rf_start_pos + NET_nolrn_info.receptiveFieldSize(:, end) - 1;

TOP = max(1 - rf_start_pos(1), 0);
LEFT = max(1 - rf_start_pos(2), 0);
template_t = padarray(template, [TOP, LEFT], 0, 'pre');
img = padarray(img, [TOP, LEFT, 0], 0, 'pre');
rf_start_pos = rf_start_pos + [TOP; LEFT];
rf_end_pos = rf_end_pos + [TOP; LEFT];

BOTTOM = max(rf_end_pos(1) - NET.meta.normalization.imageSize(1), 0);
RIGHT = max(rf_end_pos(2) - NET.meta.normalization.imageSize(2), 0);
template_t = padarray(template_t, [BOTTOM, RIGHT], 0, 'post');
img = padarray(img, [BOTTOM, RIGHT, 0], 0, 'post');

viz = vl_imsc(template_t(rf_start_pos(1):rf_end_pos(1), ...
                       rf_start_pos(2):rf_end_pos(2), :));                   
img = uint8(img(rf_start_pos(1):rf_end_pos(1), ...
                       rf_start_pos(2):rf_end_pos(2), :));

end
