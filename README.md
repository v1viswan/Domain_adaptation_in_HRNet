# Using Domain Adaptation on HRNet for a private dataset
This work is based on modifying the original HR Net from https://github.com/HRNet/HRNet-Semantic-Segmentation and using the domain adaptation based on Adversarial discriminator from the paper : Adversarial Discriminative Domain Adaptation (CVPR 2017) https://arxiv.org/abs/1702.05464 <br>


## Using the tools
Before running the scripts, please download the Cityscapes dataset as well as a pretrained model from HRNet github site and store it as mentioned in the HR Net github page. The pretrained model has to be in main folder and the file name adjusted in config file (explained below). The file was too large to be pushed to github. <br>
The file ./tools/train_vj.ipynb is an ipython notebook which was used as test place to check python code. This is my roughwork and so its confusing and not neat. <br>
./tools/tools_vj.ipynb contain code to take a video and store the frames as images. The domain adaptation can then work on these frame images. <br>

./script_train_domain_adaptation.py is the file which can be used to train the HR Net to adapt to new domain. The images of new domain should be in './domain2_images' folder or you can change the folder location at the config file experiments/cityscapes/vj_domain_adapt.yaml, where there is a comment for the section added by me. <br>

To run the training, run the command : "python tools/script_train_domain_adaptation.py --cfg experiments/cityscapes/vj_domain_adapt.yaml" from the main HR Net folder. <br>

To visualize the results, please refer ./tools/visualize_adapted_network.ipynb. You might have to adjust the saved model paths and image paths if you made changes to them in the config.


## Reference
[1] Deep High-Resolution Representation Learning for Human Pose Estimation. Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. CVPR 2019. [download](https://arxiv.org/pdf/1902.09212.pdf)
[2] Adversarial Discriminative Domain Adaptation. Tzeng, Eric and Hoffman, Judy and Saenko, Kate and Darrell, Trevor. CVPR 2017.
