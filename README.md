# SpTFuse
This is official Pytorch implementation of "SAM-guided multi-level collaborative Transformer for infrared and visible image fusion"

# Framework
The overall framework of the proposed SpTFuse.
![image](https://github.com/lxq-jnu/SpTFuse/blob/master/images/framework.png)

## Recommended Environment
 - [ ] torch  1.13.0
 - [ ] torchvision 0.14.0
 - [ ] kornia 0.7.0
 - [ ] pillow  9.4.0
 - [ ] numpy 1.21.2

# To Train

Run ```python train.py``` to train the model.

Download the SAM model from [sam](https://pan.baidu.com/s/1ARi3yGOQk5kch3mKCMukiA?pwd=p24w) and put it into `'./mobile_sam/weights/'`.

Download the training dataset from [MSRS](https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS).

# To Test

Download the checkpoint from [best_model](https://pan.baidu.com/s/1W1xBE89vY4WMgPldtVBM0g?pwd=om94) and put it into `'./checkpoint/'`.

Run `test.py`.
