function res =  salseg(im0, mask, variant)

if nargin < 3, variant = 'sal' ; end

res.mask = mask ;

opts.bVis = true ;
switch lower(variant)
  case 'ks'
    opts.quantMax = 0.95;
    opts.quantMin = 0.3;
    opts.quantMaxClamp = 0.995;
    opts.quantMinClamp = 0.005;
    opts.ksCompatible = false ;

  case {'sal'}
    opts.quantMax = 0.90;
    opts.quantMin = 0.30;
    opts.quantMaxClamp = 0.995 ;
    opts.quantMinClamp = 0.005 ;
    opts.getLargestCC = true ;
    opts.gmmNmix_fg = 5 ;
    opts.ksCompatible = false ;

  case {'am', 'dc'}
    opts.quantMax = 0.90;
    opts.quantMin = 0.30;
    opts.quantMaxClamp = 0.995 ;
    opts.quantMinClamp = 0.005 ;
    opts.getLargestCC = true ;
    opts.gmmNmix_fg = 5 ;
    opts.ksCompatible = false ;

  case {'baseline1'}
    opts.quantMax = 0.99999;
    opts.quantMin = 0.00001;
    opts.quantMaxClamp = 0.99999;
    opts.quantMinClamp = 0.00001;
    opts.getLargestCC = true ;
    opts.gmmNmix_fg = 5 ;
    opts.ksCompatible = false ;
    
  case {'baseline2'}
    idx_high = find(mask > 0.6);  
    idx_low = find(mask < 0.4);
    opts.quantMax = 1 - numel(idx_high) / numel(mask);
    opts.quantMin = numel(idx_high) / numel(mask);
    opts.quantMaxClamp = opts.quantMax;
    opts.quantMinClamp = opts.quantMin;
    opts.getLargestCC = true ;
    opts.gmmNmix_fg = 5 ;
    opts.ksCompatible = false ;

  case {'baseline3'}
    opts.quantMax = 0.90;
    opts.quantMin = 0.30;
    opts.quantMaxClamp = 0.995 ;
    opts.quantMinClamp = 0.005 ;
    opts.getLargestCC = true ;
    opts.gmmNmix_fg = 5 ;
    opts.ksCompatible = false ;

end

[res.seg, res.segBox] = ksseg(im0, res.mask, opts) ;

figure(2) ; clf ;
subplot(2,2,1) ; imagesc(im0) ; axis image ;
hold on ;
rectangle('Position', [res.segBox(1:2); res.segBox(3:4) - res.segBox(1:2) + 1], ...
          'EdgeColor', 'yellow', 'LineWidth', 2) ;
subplot(2,2,2) ; imagesc(res.mask) ; axis image ;
subplot(2,2,3) ; imagesc(res.seg) ; axis image ;
subplot(2,2,4) ; imagesc(bsxfun(@times, single(res.seg), im0)) ; axis image ;
