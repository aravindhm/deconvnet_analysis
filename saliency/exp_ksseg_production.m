function exps = exp_ksseg_production(hackLRN, prefix, method, network)
% Code to perform segmentation
% hackLRN is a boolean (True/False) on whether or not to use Identity
% instead of LRN^{BP}. If True then Identity is used in the reversed
% architecture 
% prefix - results will be dumped in the folder data/ferrari/[prefix]
% method - 'am' for DeSaliNet, 'ks' for Karen's Saliency and baseline3 for
% the gaussian baseline, 'dc' for DeconvNets
% network - 'alex' for alexnet and 'vgg-vd-16' for VGG-verydeep-16
% 
% The return value is not useful. It is simply the list of experiments -
% one for each image.
%
% Pre-requisites
% data/ferrari should contain imdb.mat and images/ - the imdb and images
% ../models/imagenet-caffe-alex.mat should contain the alexnet model
% ../models/imagenet-vgg-verydeep-16.mat should contain the VGG-verydeep-16
% model
%
% Author: Aravindh Mahendran (Copyright 2016-17)
% University of Oxford

subset = [];
opts.parallel = true;

opts.prefix = prefix;

opts.method = method;

opts.expDir = fullfile('data/ferrari/', opts.prefix) ;

imdb = load('data/ferrari/imdb.mat') ;
mkdir(opts.expDir) ;

exps = {} ;
for i = 1:numel(imdb.images.name)
  name = imdb.images.name{i} ;
  exp.imagePath = sprintf(imdb.paths.image, name) ;
  exp.resPath = sprintf('%s/%s.mat', opts.expDir, name) ;
  exp.fig1Path = sprintf('%s/%s-1.pdf', opts.expDir, name) ;
  exp.fig2Path = sprintf('%s/%s-2.pdf', opts.expDir, name) ;
  exp.fig3Path = sprintf('%s/%s-3.pdf', opts.expDir, name) ;


  exp.network = network;

  exp.alreadyDone = exist(exp.resPath,'file') ;
  exps{end+1} = exp ;
end

if isempty(subset)
  subset = 1:numel(imdb.images.name) ;
end
numel(subset)

switch exp.network
  case 'alex'
    net = dagnn.DagNN.fromSimpleNN(...
      load('../models/imagenet-caffe-alex.mat'), ...
      'canonicalNames', true) ;

  case 'vgg-vd-16'
    net = dagnn.DagNN.fromSimpleNN(...
      load('../models/imagenet-vgg-verydeep-16.mat'), ...
      'canonicalNames', true) ;
end

switch opts.method
    case 'ks'
        pred0val = 1;

    case 'am'
        pred0val = 1;
        opts.hackType = 'us';
        layerNames = {net.layers.name} ;
        for l = 1:numel(layerNames)
            li = net.getLayerIndex(layerNames{l}) ;
            if isa(net.layers(li).block, 'dagnn.ReLU')
              net.addLayer([layerNames{l} 'hacky'], ...
                   HackyReLU('hackType', opts.hackType), ...
                   net.layers(li).inputs, ...
                   net.layers(li).outputs, ...
                   {}) ;
              net.removeLayer(layerNames{l}) ;
            elseif isa(net.layers(li).block, 'dagnn.LRN') && hackLRN
              t = HackyLRN();
              t.param = net.layers(li).block.param;
              net.addLayer([layerNames{l} 'hacky'], t, ...
                   net.layers(li).inputs, ...
                   net.layers(li).outputs, ...
                   {}) ;
              net.removeLayer(layerNames{l}) ;
            end
        end

    case 'dc'
        pred0val = 1;
        opts.hackType = 'deconvnet';
        layerNames = {net.layers.name} ;
        for l = 1:numel(layerNames)
            li = net.getLayerIndex(layerNames{l}) ;
            if isa(net.layers(li).block, 'dagnn.ReLU')
              net.addLayer([layerNames{l} 'hacky'], ...
                   HackyReLU('hackType', opts.hackType), ...
                   net.layers(li).inputs, ...
                   net.layers(li).outputs, ...
                   {}) ;
              net.removeLayer(layerNames{l}) ;
            elseif isa(net.layers(li).block, 'dagnn.LRN') && hackLRN
              t = HackyLRN();
              t.param = net.layers(li).block.param;
              net.addLayer([layerNames{l} 'hacky'], t, ...
                   net.layers(li).inputs, ...
                   net.layers(li).outputs, ...
                   {}) ;
              net.removeLayer(layerNames{l}) ;
            end
        end
  
    otherwise
      pred0val = nan;
      net =struct();
end

if(opts.parallel)
 ' Using parallel for'
  parfor i = subset
    ts = tic;
    im0 = imread(exps{i}.imagePath) ;
    fprintf(1, '%s\n', exps{i}.imagePath);
    [im0, scalingFactor] = ksresize(im0);

    switch opts.method
      case {'ks', 'am', 'dc'}
         [mask, mask_signed] = kssaliency_fcn_proper(net, im0, pred0val) ;
      case 'baseline3'
         [mask, mask_signed] = baseline3(im0);
      otherwise
         assert(false) ;
    end

    res = salseg(im0/255, mask, opts.method) ;
    res.mask_signed = mask_signed;

    dothesave(exps{i}, '-struct', res) ;
    %print(1,'-dpdf', exps{i}.fig1Path) ;
    %print(2,'-dpdf', exps{i}.fig2Path) ;
    %print(101,'-dpdf', exps{i}.fig3Path) ;

    toc(ts);
  end
else
  for i = subset
    ts = tic;
    im0 = imread(exps{i}.imagePath) ;
    fprintf(1, '%d %s\n', i, exps{i}.imagePath);
    [im0, scalingFactor] = ksresize(im0);

    switch opts.method
      case {'ks','am', 'dc'}
        [mask, mask_signed] = kssaliency_fcn_proper(net, im0, pred0val) ;
      case {'baseline1'}
        mask = baseline1(im0);
      case {'baseline2'}
        mask = baseline2(im0);
      case 'baseline3'
         [mask, mask_signed] = baseline3(im0);
      otherwise
        assert(false) ;
    end
    res = salseg(im0/255, mask, opts.method) ;
    res.mask_signed = mask_signed;

    save(exps{i}.resPath, '-struct', 'res') ;
    %print(1,'-dpdf', exps{i}.fig1Path) ;
    print(2,'-dpdf', exps{i}.fig2Path) ;
    print(101,'-dpdf', exps{i}.fig3Path) ;

    toc(ts);
  end
end

function dothesave(a,b,res)
  save(a.resPath,b,'res');
  print(2,'-dpdf',a.fig2Path);
  print(101,'-dpdf',a.fig3Path);
