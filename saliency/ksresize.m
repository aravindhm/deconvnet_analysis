function [im, sc] = ksresize(im)
sz = [size(im,1) size(im,2)] ;
sc = 256/min(sz) ;
sz = round(sc*sz) ;
im = imresize(im, sz, 'bicubic') ;
im = single(im) ;
