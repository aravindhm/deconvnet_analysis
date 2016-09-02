function exp_seg_unpack()

load('data/ferrari/gtsegs_ijcv.mat','value') ;
mkdir('data/ferrari/images') ;
mkdir('data/ferrari/segs') ;

imdb.paths.image = 'data/ferrari/images/%s.jpg' ;
imdb.paths.seg = 'data/ferrari/segs/%s-v%d.png' ;
imdb.images.name = value.id ;
imdb.images.numSegs = zeros(size(imdb.images.name)) ;
[imdb.classes.name,~,imdb.images.class] = unique(value.target) ;

for i = 1:numel(value.img)
  imagePath = sprintf(imdb.paths.image, imdb.images.name{i}) ;
  imwrite(value.img{i}, imagePath) ;
  imdb.images.numSegs(i) = numel(value.gt{i}) ;

  for j = 1:imdb.images.numSegs(i)
    segPath = sprintf(imdb.paths.seg, imdb.images.name{i}, j) ;
    imwrite(value.gt{i}{j}, segPath) ;
  end
end

save('data/ferrari/imdb.mat', '-struct', 'imdb') ;
