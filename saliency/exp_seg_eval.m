function exp_seg_eval(prefix, subset, silent)
% evaluates segmentation results dumped in data/ferrari/[prefix] folder
% subset is useful to choose a subset of the examples for evaluation
% silent - true if you don't want to output anything into the terminal.

if(~exist('silent', 'var'))
  silent = false;
end

opts.prefix = prefix; 

opts.expDir = fullfile('data/ferrari/', opts.prefix) ;

imdb = load('data/ferrari/imdb.mat') ;

if(nargin < 2 || isempty(subset))
  subset = 1:numel(imdb.images.name);
end

imdb.images.class = imdb.image.class ;
K = numel(imdb.classes.name) ;
confusion = zeros(K) ;

acc = zeros(1,K) ;
iou = zeros(1,K) ;
precision = zeros(1,K);
mass = zeros(1,K) ;
found = 0;

for idx = 1:numel(subset)

  i = subset(idx);

  name = imdb.images.name{i} ;

  exp.imagePath = sprintf(imdb.paths.image, name) ;
  exp.resPath = sprintf('%s/%s.mat', opts.expDir, name) ;

  if exist(exp.resPath)
    found= 1;
    res = load(exp.resPath) ;
    [acc_, iou_, precision_] = seg_eval(imdb, i, res.seg) ;
    c = imdb.images.class(i) ;
    acc(c) = (mass(c) * acc(c) + acc_) / (mass(c) + 1) ;
    iou(c) = (mass(c) * iou(c) + iou_) / (mass(c) + 1) ;
    precision(c) = (mass(c) * precision(c) + precision_) / (mass(c) + 1) ;
    mass(c) = mass(c) + 1 ;
  else
    found = 0;
    acc_ = 0 ;
    iou_ = 0 ;
    precision_ = 0;
  end

if(~silent)
if(found)
  fprintf('%05d, %05d, %8.2f, %8.2f, %8.2f, %8.2f, %8.2f, %8.2f, %s %s\n', ...
          i, numel(imdb.images.name), ...
          mean(acc(mass>0)), acc_, mean(iou(mass>0)), iou_, ...
          mean(precision(mass>0)), precision_, ...
          name, ' ') ;
else
  fprintf('%05d, %05d, %8.2f, %8.2f, %8.2f, %8.2f, %8.2f, %8.2f, %s %s\n', ...
          i, numel(imdb.images.name), ...
          mean(acc(mass>0)), acc_, mean(iou(mass>0)), iou_, ...
          mean(precision(mass>0)), precision_, ...
          name, 'not found') ;
end
end

end

fprintf(1, '%s %8.2f, %8.2f\n', prefix, mean(acc(mass > 0)), mean(iou(mass>0)));
