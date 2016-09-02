% generates fig. 7 in the paper
IMAGE_IDS = [177, 249];

FIGS_PATH = 'genfigs/';

IMAGE_NAMES = {sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(1)), ...
    sprintf('imagenet12-val/ILSVRC2012_val_%08d.JPEG', IMAGE_IDS(2))};
     
im = imread(IMAGE_NAMES{1}) ;
im = rgb2gray(im2single(im)) ;
im = resizencrop(im, [224,224]) ;

im_2 = imread(IMAGE_NAMES{2}) ;
im_2 = rgb2gray(im2single(im_2)) ;
im_2 = resizencrop(im_2, [224,224]) ;

% for real images one muust have
% Fm1 == circshift(flipud(fliplr(Fm1)),[1 1]) !

F = fft2(im) ;
Fa = angle(F) ;
Fm = abs(F) ;
Fm1 = randn(size(Fm)) ;
%Fm1 = Fm1 + circshift(flipud(fliplr(Fm1)),[1 1]) ;
%Fm1 = Fm1*0 + 1 ;
Fm2 = abs(Fm1) ;

%F_other = fft2(im_2);
%Fm2 = abs(F_other); % Get the magnitude from another image.

im0 = real(ifft2(Fm .* exp(1i*Fa))) ;
im1 = real(ifft2(Fm1 .* exp(1i*Fa))) ;
im2 = real(ifft2(Fm2 .* exp(1i*Fa))) ;

figure ; clf;
subplot(1,3,1) ; imagesc(im0) ; axis image ; axis off; title('Original Amplitude');
subplot(1,3,2) ; imagesc(im1) ; axis image ; axis off; title('Randomized Amplitude');
subplot(1,3,3) ; imagesc(im2) ; axis image ; axis off; title('Positive Randomized Amplitude') ;

im0_viz = padarray(im0, [1,1], 1, 'both');
im1_viz = padarray(im1, [1,1], 1, 'both');
im2_viz = padarray(vl_imsc_am(im2), [1,1], 1, 'both');

% Now to visualize VGG-16 fc8 for this image.
opts = struct();
opts.randomizeWeights = false;
opts.gpu = false;

opts.relus_to_change = 1:100;
opts.pools_to_change = 1:5;
opts.convs_to_change = 1:100;

opts.neuron_I = 1;
opts.neuron_J = 1;
opts.neuron_channel = inf;

opts.imagePath = IMAGE_NAMES{1};
opts.modelPath = 'models/imagenet-vgg-verydeep-16.mat';
opts.layer = 36;
opts.algorithm = 'TTT';
[~, TTT_viz, opts_new, NET, ~] = hand_specified_neuron_viz_fn(opts);
TTT_viz = mean(padarray(vl_imsc_am(TTT_viz), [1,1], 1, 'both'), 3);

%im_2_pp = opts_new.normalize(rgb2gray(imread(IMAGE_NAMES{2})));
%res = vl_simplenn(NET, im_2_pp);
%Fm2 = res(end).x;

im_1_pp = opts_new.normalize(rgb2gray(imread(IMAGE_NAMES{1})));
res = vl_simplenn(NET, im_1_pp);
Fm1 = res(end).x;

res2 = vl_simplenn(NET, im_1_pp, abs(randn(size(Fm1), 'single')));
TTT2_viz = mean(padarray(vl_imsc_am(res2(1).dzdx), [1, 1], 1, 'both'), 3);

clear NET;

% Now to visualize VGG-16 fc8 for this image using deconvnet instead of TTT.
opts = struct();
opts.randomizeWeights = false;
opts.gpu = false;

opts.relus_to_change = 1:100;
opts.pools_to_change = 1:5;
opts.convs_to_change = 1:100;

opts.neuron_I = 1;
opts.neuron_J = 1;
opts.neuron_channel = inf;

opts.imagePath = IMAGE_NAMES{1};
opts.modelPath = [MATCONVNET_PATH, 'data/models/imagenet-vgg-verydeep-16.mat'];
opts.layer = 36;
opts.algorithm = 'deconvnet';
[~, deconvnet_viz, opts_new, NET, ~] = hand_specified_neuron_viz_fn(opts);
deconvnet_viz = mean(padarray(vl_imsc_am(deconvnet_viz), [1,1], 1, 'both'), 3);

%im_2_pp = opts_new.normalize(rgb2gray(imread(IMAGE_NAMES{2})));
%res = vl_simplenn(NET, im_2_pp);
%Fm2 = res(end).x;

res2 = vl_simplenn(NET, im_1_pp, abs(randn(size(Fm1), 'single')));
deconvnet2_viz = mean(padarray(vl_imsc_am(res2(1).dzdx), [1, 1], 1, 'both'), 3);



%opts.algorithm = 'deconvnet';
%[~, DeConvNet_viz, ~, ~, ~] = hand_specified_neuron_viz_fn(opts);
%DeConvNet_viz = mean(padarray(vl_imsc_am(DeConvNet_viz), [1,1], 1, 'both'), 3);

%opts.algorithm = 'saliency';
%[~, Saliency_viz, ~, ~, ~] = hand_specified_neuron_viz_fn(opts);
%Saliency_viz = mean(padarray(vl_imsc_am(Saliency_viz), [1,1], 1, 'both'), 3);

%final_image = [padarray(im,[1,1],0,'both'), im0_viz, im1_viz, im2_viz;...
%              zeros(size(im) + 2), DeConvNet_viz, Saliency_viz, TTT_viz];
final_image = [padarray(im,[1,1],1,'both'), im0_viz, im2_viz, TTT2_viz, deconvnet2_viz];

imwrite(final_image, [FIGS_PATH, 'fourier.png']);