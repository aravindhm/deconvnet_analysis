# deconvnet_analysis
Code for "Salient Deconvolutional Networks, Aravindh Mahendran, Andrea Vedaldi, ECCV 2016"

### Parts of this code

1. Generate figures in the paper
2. Segmentation code for table 1 and .mat files required to generate figure 8.
3. Supplementary material figures.

Parts 1 and 2 are complete and documented. Part 3 is mainly an extension of part 1 with more images

### How to run this code
Follow these steps. ROOT refers to the main directory containing the file generate_all_figures.m

1. Copy and install matconvnet into ROOT/matconvnet . Delete ROOT/matconvnet/matlab/simplenn/vl_simplenn.m
2. Copy vlfeat into ROOT/vlfeat
3. Download the alexnet model - imagenet-caffe-alex from http://www.vlfeat.org/matconvnet/pretrained/ into ROOT/models
4. Similarly download the vgg-verydeep-16 model - imagenet-vgg-verydeep-16 into ROOT/models
5. Download the imagenet validation dataset into ROOT/imagenet12-val .
6. Download and install gsc-1.2 from https://www.robots.ox.ac.uk/~vgg/software/iseg/ into ROOT/saliency/gsc
7. Download gtsegs_ijcv.mat from http://groups.inf.ed.ac.uk/calvin/proj-imagenet/data/ into saliency/data/ferrari
8. Start matlab and change directory to ROOT/saliency. Run exp_seg_unpack() - this will unpack gtsegs_ijcv.mat and compute the imdb.mat for the Ferrari dataset [4].
9. Explore ROOT/saliency/run_2.m to run different segmentation experiments to get numbers in Table 1 of the paper.
10. Explore and run ROOT/generate_all_figures.m to get figures from the paper. They will be saved into the ROOT/genfigs folder.

Note that step 9 assumes all the experiments for segmentation in ROOT/saliency/run_2.m were run. This assumption applies only when generating figure 8. You can also uncomment segmentation_qualitative_results_figure to skip it.

### Research code
If anything doesn't work then please post the issue and I'll try to fix it. This is research code and comes with no WARRANTY or GUARANTY of any sort.

### References
1. Guillaumin, M., Küttel, D., Ferrari, V.: Imagenet auto-annotation with segmentation propagation. In: IJCV (2014)
2. Noh, H., Hong, S., Han, B.: Learning deconvolution network for semantic segmentation. In: Proc. ICCV (2015)
3. Simonyan, K., Vedaldi, A., Zisserman, A.: Deep inside convolutional networks: Visualising image classification models and saliency maps. In: ICLR (2014)
4. Springenberg, J.T., Dosovitskiy, A., Brox, T., Riedmiller, M.: Striving for simplicity: The all convolutional net. In: ICLR Workshop (2015)
5. Zeiler, M.D., Fergus, R.: Visualizing and understanding convolutional networks. In: Proc. ECCV (2014)
6. V. Gulshan, C. Rother, A. Criminisi, A. Blake and A. Zisserman.: Geodesic star convexity for interactive image segmentation. In: Proc. CVPR (2010)
