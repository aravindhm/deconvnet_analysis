function [mask, mask_signed] = baseline3(im0)

sz = size(im0);
mask = zeros(sz(1:2), 'single');

f = fspecial('gaussian', [125, 125], 24);
t = 62;
mask(end/2 - t:end/2 + t, end/2 - t:end/2 + t) = ...
  mask(end/2 - t:end/2 + t, end/2 - t:end/2 + t) + f;

t = 63;
mask(1:t,1:t) = -f(t:end, t:end);
mask(end-t+1:end,1:t) = -f(1:t, t:end);
mask(end-t+1:end,end-t+1:end) = -f(1:t,1:t);
mask(1:t,end-t+1:end) = -f(t:end,1:t);

mask = vl_imsc(mask);
mask_signed = mask;

end % end mask = baseline3(im0);
