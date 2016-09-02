% We need to call the following function with different inputs to complete
% all the experiments
% function exps = exp_ksseg_production(hackLRN, prefix, method, network)
% Uncomment different lines in the code below to get different parts of table 1.
%
% exp_ksseg_production runs the segmentation code and dumps the results
% into data/ferrari/prefix where prefix is one of the input arguments to
% this function
%
% exp_seg_eval - evaluates the segmentation results dumped above and
% outputs 
% prefix Per-pixel-accuracy and 
%
% pre-requisites 
% gsc contains the gsc-1.2 from - http://www.robots.ox.ac.uk/~vgg/research/iseg/
% + pre-requisites in exp_ksseg_production.m
% + pre-requisites in exp_seg_eval.m
% 
% Author: Aravindh Mahendran (Copyright 2016-17)
% University of Oxford







run gsc/setup.m

%exps = exp_ksseg_production(true, 'sal-dc-101-vd'  , 'dc', 'vgg-vd-16');
%exps = exp_ksseg_production(false, 'sal-ks-101-alex', 'ks', 'alex');
%exps = exp_ksseg_production(false, 'sal-ks-101-vd'  , 'ks', 'vgg-vd-16');
exps = exp_ksseg_production(false, 'sal-am-101-alex', 'am', 'alex');
%exps = exp_ksseg_production(false, 'sal-am-101-vd'  , 'am', 'vgg-vd-16');
%exps = exp_ksseg_production(false, 'sal-baseline3-101', 'baseline3', '');

%exp_seg_eval('sal-ks-101-alex', [], true);
%exp_seg_eval('sal-ks-101-vd', [], true);

%exp_seg_eval('sal-am-101-alex', [], true);
exp_seg_eval('sal-am-101-alex', [], true);
%exp_seg_eval('sal-am-101-vd', [], true);

%exp_seg_eval('sal-baseline3-101', [], true);
