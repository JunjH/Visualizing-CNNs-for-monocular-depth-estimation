# Visualization of Convolutional Neural Networks for Monocular Depth Estimation
<br>

Junjie Hu, Yan Zhang, Takayuki Okatani, "Visualization of Convolutional Neural Networks
for Monocular Depth Estimation," ICCV, 2019. [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Visualization_of_Convolutional_Neural_Networks_for_Monocular_Depth_Estimation_ICCV_2019_paper.pdf)

Introduction
-
We attempt to interpret CNNs for monocular depth estimation. To this end, we propose to locate the most relevant pixels of input image to depth inference. We formulate it as an optimization problem of identifying the smallest number of image pixels from which the CNN can estimate a depth map with the minimum difference from the estimate from the entire image. 

![](https://github.com/junjH/Visualizing-CNNs-for-monocular-depth-estimation/raw/master/figs/fig_arch.png)

Predicted Masks
-
![](https://github.com/junjH/Visualizing-CNNs-for-monocular-depth-estimation/raw/master/figs/fig_mask.png)

Extensive experimental results show

+ The behaviour of CNNs that they seem to select edges in input images depending not on their strengths but on importance for inference of scene geometry.

+ The tendency of attending not only on the boundary but the inside region of each individual object.

+ The importance of image regions around the vanishing points for depth estimation on outdoor scenes.


Please check our paper for more details.

Dependencies
-
+ python 2.7<br>
+ pytorch 0.3.1<br>

Running
-

Download the trained networks for depth estimation : [Depth estimation networks](https://drive.google.com/file/d/1QaUkdOiGpMuzMeWCGbey0sT0wXY0xtsj/view?usp=sharing) <br>

Download the trained networks for mask prediction : [Mask prediction network](https://drive.google.com/file/d/12VXcfSEZ_Te13w4WYJyoaJjkhUxOEqwo/view?usp=sharing) <br>

Download the NYU-v2 dataset: [NYU-v2 dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) <br>

+ ### Test<br>
  python test.py<br>
+ ### Train<br>
  python train.py<br>

Citation
-
If you use the code or the pre-processed data, please cite:

    @inproceedings{Hu2019VisualizationOC,
      title={Visualization of Convolutional Neural Networks for Monocular Depth Estimation},
      author={Junjie Hu and Yan Zhang and Takayuki Okatani},
      booktitle={IEEE International Conf. on Computer Vision (ICCV)},
      year={2019}
    }
    
    @inproceedings{Hu2018RevisitingSI,
      title={Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps With Accurate Object Boundaries},
      author={Junjie Hu and Mete Ozay and Yan Zhang and Takayuki Okatani},
      booktitle={IEEE Winter Conf. on Applications of Computer Vision (WACV)},
      year={2019}
    }
