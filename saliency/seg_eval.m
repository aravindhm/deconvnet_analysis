function [acc, iou, precision] = seg_eval(imdb, i, seg, doplot)

if nargin < 4, doplot = false ; end

m = imdb.images.numSegs(i) ;
precision_ = zeros(1,m) ;
acc_ = zeros(1,m) ;
iou_ = zeros(1,m) ;
for j = 1:m
  gt{j} = imread(sprintf(imdb.paths.seg, imdb.images.name{i}, j)) ;
  gt{j} = uint8(gt{j}) ;
  if j == 1
    mass_ = numel(gt{j}) ;
    est = imresize(seg, size(gt{j}), 'nearest') ;
  end
  acc_(j) = sum(gt{j}(:) == est(:)) / mass_ * 100 ;
  iou_(j) = sum(gt{j}(:) > 0 & est(:) > 0) / sum(gt{j}(:) + est(:) > 0) * 100 ;
  precision_(j) = sum(gt{j}(:) > 0 & est(:) > 0) / sum(est(:) > 0);
end
[acc,j] = max(acc_) ;
[iou,j] = max(iou_) ;
[precision,j] = max(precision_);

if doplot
  figure(200) ; clf ;
  imagesc([est > 0, gt{j} > 0]) ;
  drawnow ;
end
