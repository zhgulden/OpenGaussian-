# MV-OpenGaussian

## Acknowledgements

This repository is a modification of original [OpenGaussian](https://github.com/yanmin-wu/OpenGaussian) project.

## Installation

The installation of MV-OpenGaussian is almost same to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```
git clone https://github.com/zhgulden/multiview-opengaussian.git
```
Then install the dependencies:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting

# the rasterization lib comes from DreamGaussian
cd OpenGaussian/submodules
unzip ashawkey-diff-gaussian-rasterization.zip
pip install ./ashawkey-diff-gaussian-rasterization
```
+ other additional dependencies: bitarray, scipy, pytorch3d
    ```shell
    pip install bitarray scipy
    
    pip install git+https://github.com/facebookresearch/pytorch3d
    ```
+ `simple-knn` is not required

## About modifications

### CLIP feature multi-view enhancement

M1 and M2 modifications can be found in `train_M1.py` and `train_M2.py`.

### Multi-view SAM refinement

Modifications are located in `utils/sam_refinement_utils.py` and the respective launch lines are included in each of `train_baseline.py`, `train_M1.py` and `train_M2.py` and can be activated by setting `--enable_multiview_sam_refinement` in `train_scannet.sh`

**NOTE:** to visualize refinement stages, set `VISUALIZE_STAGE_1`/`VISUALIZE_STAGE_2` to `True` [here](utils/sam_refinement_utils.py).
