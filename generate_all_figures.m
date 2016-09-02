% This script calls all the other scripts to generate all the images in the
% paper
% Author: Aravindh Mahendran (Copyright 2016-17)
% University of Oxford

clear all; setup_eccv2016code;
'fig splash'
fig_splash_all_images; % fig. 1


clear all; setup_eccv2016code;
'Spatial selectivity no arch effects'
fig_spatial_selectivity_noarcheffects; % fig. 6


clear all; setup_eccv2016code;
'effect of phase deconvnet'
effect_of_phase_deconvnet; % fig. 4


clear all; setup_eccv2016code;
'components_coding_phase_info_figure'
components_coding_phase_info_figure; % fig. 3


clear all; setup_eccv2016code;
'segmentation_qualitative_results_figure'
segmentation_qualitative_results_figure; % fig. 8


clear all; setup_eccv2016code;
'effect_of_architecture_figure'
effect_of_architecture_figure; % fig. 5


clear all; setup_eccv2016code;
'effect_of_lrn_nobackprop'
effect_of_lrn_nobackprop; % supplementary material fig. 1


clear all; setup_eccv2016code;
'fourier proper'
fourier_phase; % fig. 7