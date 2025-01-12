# SHA_GAN
Code for our paper "[SHA_GAN] 

### Introduction


### Requirements

1. Tensorflow = 1.12
2. Python 3
3. NVIDIA GPU + CUDA 9.0
4. Tensorboard


### Installation

1. Clone this repository

   ```bash
   git clone https://github.com/IPCSRG/SHA_GAN
   ```
   
### Running

**1.   Datasets**

We train our model on Places2 and CelebA-HQ datasets.

1. [Places2](http://places2.csail.mit.edu)
2. [CelebA-HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 

generate the image list using script  [`./get_flist.py`](./get_flist.py) for training.

**2.   Training**

To train our model, modify the model config file [inpaint.yaml](inpaint.yaml). You may need to change the path of dataset or the parameters of the networks etc. Then run python train.py \

**3.   Testing**

To output the generated results of the inputs, you can use the [multitest.py](multitest.py).  The pre-trained weights can be downloaded from [Places2](), [CelebA-HQ](). Download the checkpoints and save them to './model_logs'

### Citation

We built our code based on [CA](https://github.com/JiahuiYu/generative_inpainting).

```
@article{Han18,
  author  = {Han Zhang, Ian J. Goodfellow, Dimitris N. Metaxas, Augustus Odena},
  title   = {Self-Attention Generative Adversarial Networks},
  year    = {2019},
  journal = {International Conference on Machine Learning},
}
```


### Acknowledgements

We built our code based on [CA](https://github.com/JiahuiYu/generative_inpainting). Please consider to cite their papers. 
