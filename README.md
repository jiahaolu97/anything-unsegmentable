# Anything Unsegmentable

This is a PyTorch implementation of our paper "Unsegment Anything by Simulating Deformation" (CVPR 2024).

## Installation

Create a conda environment for the project. The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.


You can either install the conda environment through the yml file:
```shell
conda env create -f unsegment_env.yml
```

or you would prefer custom your installation by:

```shell
conda create --name unsegment python=3.9 -y
conda activate unsegment
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia

#Install environments required by Segment Anything and FastSAM:
conda install matplotlib PyYAML tqdm requests
pip install pycocotools opencv-python ultralytics==8.0.120
pip install git+https://github.com/openai/CLIP.git

#Other tools for developing anything-unsegmentable
pip install pytorch-msssim, jupyter
```

## <a name="GettingStarted"></a>Getting Started

Segment Anything dataset can be downloaded [here](https://ai.meta.com/datasets/segment-anything-downloads/). By downloading the datasets you agree that you have read and accepted the terms of the SA-1B Dataset Research License. Experiments of the paper were conducted on the subset `sa_000000.tar`. 


Download Segment Anything and FastSAM model checkpoints:
- `SAM_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- `SAM_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `SAM_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `FastSAM`: [YOLOv8x based Segment Anything Model](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing) | [Baidu Cloud (pwd: 0000).]


Change the directory path to yours in `config.py`:
```
SAM_DATASET_PATH = 'path_to_your_folder'
SAM_MODEL_PATH = 'path_to_your_folder'
FASTSAM_MODEL_PATH = 'path_to_your_folder'
```

Benchmark different adversarial attacks on Segment Anything:
```
python exp/benchmark.py
```

Play with other visualization tools or experiments in `exp/` folder.

Develop your adversarial attacks in `attack/attacker.py`.

Interface of the UAD attack:
```
attacker.UAD(clean_cv2_image,
            control_loss_alpha=0.05, 
            fidelity_loss_alpha=1, 
            deform_epochs=40, 
            est_fidelity_iter=4,
            warp_size=4, 
            warp_filter_stride=300, 
            num_split=6, 
            src_rand_range=0.0, 
            dst_rand_range=0.1,
            use_DI=False, 
            DI_noise_std=4.0/255, 
            use_MI=False, 
            MI_momentum=1)

--control_loss_alpha: scale of control loss, $\lambda_C$
--fidelity_loss_alpha: scale of fidelity loss, $\lamda_F$
--deform_epoch: number of steps for deformation, $T_D$
--est_fidelity_iter: number of steps for proxy adversarial update, $T_f$
--warp_size: number of flow fields to be optimized in parallel
--warp_filter_stride: size of each patch for final patchwise deformation target
--num_split: number of control points for flow field
--src_rand_range: flow field control points initial starting position
--dst_rand_range: flow field control points initial ending position
--use_DI: whether to use DI(input diversity) trick
--DI_noise_std: if use_DI=True, the intensity of adding noise for input diversity
--use_MI: whether to use MI(gradient momentum) trick
--MI_momemtum: if use_MI=True, the momentum parameter for MI
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).


## Citing Our Paper

If you find this code useful for your research, please consider citing:

```
@inproceedings{lu2024unsegment,
        title={Unsegment Anything by Simulating Deformation},
        author={Lu, Jiahao and Yang, Xingyi and Wang, Xinchao},
        booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2024}
}
```
