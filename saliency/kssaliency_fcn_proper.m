function [mask, mask_signed] = kssaliency_fcn_proper(net, im0, pred0val)

im_mean = mean(mean(net.meta.normalization.averageImage,1),2) ;
im0 = bsxfun(@minus, im0, im_mean) ;

% get class
net.vars(net.getVarIndex('prediction')).precious = true ;
net.eval({'input', im0}) ;
prediction = net.vars(net.getVarIndex('prediction')).value;
[~,label] = max(squeeze(max(max(prediction, [], 1), [], 2)));

pred0 = zeros(size(prediction),'single') ;
idx = find(prediction(:,:,label) > 0);
pred0_plate = zeros(size(prediction, 1), size(prediction, 2), 'single');
pred0_plate(idx) = pred0val;
pred0(:,:,label) = pred0_plate ;

net.eval({'input', im0}, {'prediction', pred0});
dd = net.vars(net.getVarIndex('input')).der;
mask = max(abs(dd), [], 3);
mask_signed = dd;
