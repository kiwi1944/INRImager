# Physics-Informed Implicit Neural Representation for Wireless Imaging in RIS-Aided ISAC System (INRImager)

This is a PyTorch implementation of the paper "**Physics-Informed Implicit Neural Representation for Wireless Imaging in RIS-Aided ISAC System**" in *IEEE Transactions on Wireless Communications*.
Its conference version will be presented in ICC 2026, Glasgow, Scotland, UK, May 2025.
Arxiv link: 

This paper introduces **implicit neural representation** for wireless imaging and applies it to RIS-aided ISAC systems.
Specifically, physics-informed loss functions are formulated based on wireless channel models to impose physical constraints during NN training.


# Packages

- python==3.8.0
- pytorch==2.0.0
- numpy==1.24.4
- wandb==0.19.8


# Training

The training scripts come with several options.
An example for training is:

```
python train.py --wandb_project 'INRImager'
```

# Testing

Inference is performed during each training epoch.
Re-execute the inference process derives the predicted image.

# Citation

```
@article{huang2026physics,
  title={Physics-Informed Implicit Neural Representation for Wireless Imaging in {RIS}-Aided {ISAC} System},  
  author={Huang, Yixuan and Yang, Jie and Wen, Chao-Kai and Jin, Shi},
  journal={IEEE Trans. Wireless Commun.},
  year={early access, Feb. 2025},
  publisher={IEEE}
}

@inproceedings{confhuang2026physics,
  title={Physics-Informed Wireless Imaging with Implicit Neural Representation in {RIS}-Aided {ISAC} System},
  author={Huang, Yixuan and Yang, Jie and Wen, Chao-Kai and Li, Xiao and Jin, Shi},
  booktitle={Proc. Int. Conf. Commun. (ICC)},
  pages={1--6},
  year={May 2026}
}
```
