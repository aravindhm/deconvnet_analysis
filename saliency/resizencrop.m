function IMG_edited = resizencrop(IMG, imsize)
% IMG_edited = resizencrop(IMG, imsize) - resize and crop the image
%    This helper function transforms the image so as to preserve the aspect ratio before cropping off the extra bits.

% Resize preserving aspect ratio
if(size(IMG, 1) / size(IMG,2) > imsize(1) / imsize(2))
  % Image is taller than required
  IMG = imresize(IMG, [nan, imsize(2)]);
else
  % Image is broader than required
  IMG = imresize(IMG, [imsize(1), nan]);
end

% Crop off the sides so as to get it right
offset_i = floor((size(IMG, 1) - imsize(1))/2) + 1;
offset_j = floor((size(IMG, 2) - imsize(2))/2) + 1;

IMG_edited = IMG(offset_i:offset_i + imsize(1) - 1, ...
    offset_j : offset_j + imsize(2) - 1, :);

end
