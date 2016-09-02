function [seg, segBox, labels, labelsClamp] = ksseg(img, imgSal, varargin)
% I don't understand this code. Sorry :(

opts.bVis = true ;
opts.quantMax = 0.95;
opts.quantMin = 0.3;
opts.quantMaxClamp = 0.995;
opts.quantMinClamp = 0.005;
opts.ksCompatible = true ;
opts.getLargestCC = true ;
opts.gmmNmix_fg = 5;
opts.gmmNmix_bg = 5;

opts = vl_argparse(opts, varargin) ;

if opts.ksCompatible
  % karen's code assumes the saliency is a square (256 x256 with
  % accumulation?)
  imgSal = imgSal(...
    round((1:256)+(size(imgSal,1)-256)/2), ...
    round((1:256)+(size(imgSal,2)-256)/2), ...
    :) ;
end

% thresholds
TMax = quantile(imgSal(:), opts.quantMax);
TMin = quantile(imgSal(:), opts.quantMin);

useClamp = true;
if opts.quantMaxClamp <= 1
  TMaxClamp = quantile(imgSal(:), opts.quantMaxClamp);
else
  TMaxClamp = +inf ;
end
if opts.quantMinClamp >= 0
  TMinClamp = quantile(imgSal(:), opts.quantMinClamp);
else
  TMinClamp = -inf ;
end

w = size(img, 2);
h = size(img, 1);
assert(w == 256 || h == 256);
labels = zeros(h, w, 'uint8'); % by def = ignore
labelsClamp = zeros(h, w, 'uint8'); % by def = ignore

% find sub-region for which the saliency is computed
if opts.ksCompatible
  imgSalSize = 256 ;
  if w > h
    y0 = 1 + 0.5 * (256 - imgSalSize);
    x0 = 1 + 0.5 * (256 - imgSalSize) + round(0.5 * (w - 256));
  else
    x0 = 1 + 0.5 * (256 - imgSalSize);
    y0 = 1 + 0.5 * (256 - imgSalSize) + round(0.5 * (h - 256));
  end
  y1 = y0 + imgSalSize - 1;
  x1 = x0 + imgSalSize - 1;

  labelsCrop = labels(y0:y1, x0:x1);
  labelsCrop(imgSal >= TMax) = 1;
  labelsCrop(imgSal <= TMin) = 2;
  labels(y0:y1, x0:x1) = labelsCrop;

  if useClamp
    labelsCrop = labelsClamp(y0:y1, x0:x1);
    labelsCrop(imgSal >= TMaxClamp) = 1;
    labelsCrop(imgSal <= TMinClamp) = 2;
    labelsClamp(y0:y1, x0:x1) = labelsCrop;
  end
else
  % labels: 2=bg, 1=fg 0=ignore?
  labels(imgSal >= TMax) = 1;
  labels(imgSal <= TMin) = 2;
  if useClamp
    labelsClamp(imgSal >= TMaxClamp) = 1 ; % fg
    labelsClamp(imgSal <= TMinClamp) = 2 ; % bg
  end
end

% Make segmentation options
Params = bj.segOpts();
Params.gcGamma = 150;
Params.gmmNmix_fg = opts.gmmNmix_fg ;
Params.gmmNmix_bg = opts.gmmNmix_bg ;
Params.postProcess = true;

% Intialize the segmentation object
segH = bj.segEngine(0,Params);

% preProcess image
segH.preProcess(im2double(img)); % Only takes in double images

% Get the first segmentation
segH.start(labels, labelsClamp);
seg = segH.seg;

% largest connected component
seg0 = seg ;
if opts.getLargestCC
  cc = bwconncomp(seg) ;
  numPixels = cellfun(@numel, cc.PixelIdxList);
  [~, maxIdx] = max(numPixels);
  seg(:) = 0;
  seg(cc.PixelIdxList{maxIdx}) = 1 ;
end

% box
[i,j] = find(seg);
segBox = [min(j(:)); min(i(:)); max(j(:)); max(i(:))];

if opts.bVis
  figure(101) ; clf ;
  subplot(2,3,1) ; imagesc(img) ; axis image ; hold on ;
  rectangle('Position', [segBox(1:2); segBox(3:4) - segBox(1:2) + 1], 'EdgeColor', 'yellow', 'LineWidth', 2);
  subplot(2,3,2) ; imagesc(imgSal) ; axis image ; colormap gray
  subplot(2,3,3) ; imagesc(labels,[0 2]) ; axis image ; title('Soft labels') ;
  subplot(2,3,4) ; imagesc(labelsClamp,[0 2]) ; axis image ; title('Hard labels') ;
  subplot(2,3,5) ; imagesc(seg) ; axis image ;
  subplot(2,3,6) ; imagesc(seg0) ; axis image ;
end

% starts from 0, convert back to full-res
%segBox = (segBox - 1) * scalingFactor ;
