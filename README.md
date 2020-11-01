Certified Defense to Image Transformations via Randomized Smoothing <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
=============================================================================================================

We present extend an extension of randomized smoothing to cover
parametrized transformations (here image rotations and translations)
and certify robustness in the parameter space (e.g., rotation angle).
Here we evaluate three defense with different guarantees based
this. For further details, please see [our NeurIPS 2020
paper](https://www.sri.inf.ethz.ch/publications/fischer2020smoothing).

![img](https://raw.githubusercontent.com/eth-sri/transformation-smoothing/main/img.png)

## Prequisites


- a modern GNU C++ compiler (tested with 7.5.0)
- modern CUDA developer toolkit (tested with 10.1, 10.2)
- python version >= 3.6

## Install
We recommend to use conda.

``` shell
git clone --recurse-submodules https://github.com/eth-sri/transformation-smoothing.git
cd transformation-smoothing
conda create -n smooth python=3.6
conda activate smooth
conda install pytorch=1.5 torchvision cudatoolkit=YOUR_CUDA_VERSION -c pytorch
pip install -r requirements.txt
cd geometrictools
make python # installes python bindings for our C++ library; needs to be executed in the correct python environemnt for details see geometrictools/README.md
cd -
```

Make sure that `YOUR_CUDA_VERSION` is replaced your CUDA version (seen in the top right of `nvidia-smi`).
We tested this with 10.1 and 10.2.

### Datasets

Download ImageNet to the folder `./ds/imagenet` and GTSRB to
`./ds/GTSRB`.  For GTSRB we recommend to download the dataset from
<https://1drv.ms/u/s!An8jrZtDgrMljdt7o2khe7TGmZWbUg> (which already
includes a labels file) and run the file
`./ds/GTSRB/test/to_folder.py` to prepare the test set.
Other datasts will be downloaded automatically.

### Models
Training instructions for all models are given below.
The pretrained models can also be downloaded [here](https://files.sri.inf.ethz.ch/transformation-smoothing/models.tar.gz).

## Experiments
Here we provide commands to reproduce the experiments from the paper.

### Heuristic Defense (BaseSPT, cf. Section 6.1)

#### Training the Models
``` shell
python train.py --batch-size 1024 --attack false --sigmaN 0 --eps 0 --filter-size 0 --filter-sigma 0 --rotate 0 --crop-resize-crop 0 --crop-resize-resize 0 --out-dir mnist_convnet_clean_60_180 --dataset mnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0 --mnist-convnet True
python train.py --batch-size 256 --step-lr 30 --epochs 90 --sigmaN 0.0 --eps 0.0 --attack False --vingette none --filter-sigma 0 --filter-size 0 --rotate 0 --out-dir cifar_clean_30_90 --crop-resize-resize 0 --crop-resize-crop 0 --lr 0.1 --dataset cifar --start-epoch 0
python classify_f.py --model mnist_convnet_clean_60_180 --dataset mnist
python classify_f.py --model cifar_clean_30_90 --dataset cifar
```

``` shell
python classifyHeuristic.py --dataset mnist --model mnist_convnet_clean_60_180 -N 1000 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --name heuristic_rot_mnistclean_30_100_1000
python classifyHeuristic.py --dataset cifar --model cifar_clean_30_90 -N 1000 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --name heuristic_rot_cifarclean_30_100_100
python classifyHeuristic.py --dataset imagenet --model resnet50 -N 1000 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --name heuristic_rot_imagenetclean_30_100_1000

python classifyHeuristic.py --dataset mnist --model mnist_convnet_clean_60_300 -N 1000 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --transformation trans --sigma-gamma 4 --gamma 4 --name heuristic_trans_mnistclean_4_100_1000
python classifyHeuristic.py --dataset cifar --model cifar_clean_30_90 -N 1000 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --transformation trans --sigma-gamma 4 --gamma 4 --name heuristic_trans_cifarclean_4_100_1000
python classifyHeuristic.py --dataset imagenet --model resnet50 -N 1000 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --transformation trans --sigma-gamma 20 --gamma 20 --name heuristic_trans_imagenetclean_20_100_1000

python eval.py "results/classifyHeuristic/**"
```

### Distributional Defense (DistSPT, cf. Section 6.2)

#### Train Models
``` shell
python train.py --batch-size 256 --sigmaN 0.3 --eps 0.2 --filter-size 5 --filter-sigma 2.0 --rotate 30 --crop-resize-crop 0 --crop-resize-resize 0 --out-dir mnist_rot30_sigma03_eps02_60_180_filter5_20 --dataset mnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0 --crop-resize-resize 0 --crop-resize-crop 28
python train.py --batch-size 256 --sigmaN 0.3 --eps 0.2 --filter-size 5 --filter-sigma 2.0 --rotate 180 --crop-resize-crop 0 --crop-resize-resize 0 --out-dir mnist_rot180_sigma03_eps02_60_180_filter5_20 --dataset mnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0 --crop-resize-resize 0 --crop-resize-crop 28
python train.py --batch-size 256 --sigmaN 0.3 --eps 0.2 --filter-size 5 --filter-sigma 2.0 --translate 0.5 --rotate 0 --crop-resize-crop 0 --crop-resize-resize 0 --out-dir mnist_trans05_sigma03_eps02_60_180_filter5_20 --dataset mnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0 --crop-resize-resize 0 --crop-resize-crop 28

python train.py --dataset cifar --batch-size 400 --nr-worker 64 --sigmaN 0.3 --eps 1.0 --attack True --filter-sigma 1.0 --filter-size 5 --rotate 30 --epochs 90 --out-dir cifar_rot30_sigma03_eps1_filter5_10 --vingette circ --vingette-offset 2

python train.py --dataset imagenet --batch-size 400 --nr-worker 64 --sigmaN 0.5 --eps 1.0 --attack True --filter-sigma 2.0 --filter-size 5 --rotate 30 --epochs 90 --out-dir imagenet_rot30_sigma05_eps1_filter5_20 --vingette circ --vingette-offset 2

python train.py --dataset restricted_imagenet --batch-size 400 --nr-worker 64 --sigmaN 0.5 --eps 1.0 --attack True --filter-sigma 2.0 --filter-size 5 --rotate 30 --epochs 90 --out-dir rimagenet_rot30_sigma05_eps1_filter5_20 --vingette circ --vingette-offset 2

python classify_f.py --dataset mnist --model mnist_rot30_sigma03_eps02_60_180_filter5_20
python classify_f.py --dataset mnist --model mnist_rot180_sigma03_eps02_60_180_filter5_20
python classify_f.py --dataset mnist --model mnist_trans05_sigma03_eps02_60_180_filter5_20
python classify_f.py --dataset cifar --model cifar_rot30_sigma03_eps1_filter5_10
python classify_f.py --dataset imagenet --batch-size 400 --model imagenet_rot30_sigma05_eps1_filter5_20 --resize-post-transform 256 --center-crop-post-transform 224
python classify_f.py --dataset restricted_imagenet --batch-size 400 --filter-sigma 2.0 --filter-size 5 --model rimagenet_rot30_sigma05_eps1_filter5_20 --radiusDecrease 2 --resize-post-transform 256 --center-crop-post-transform 224
```

#### Obtain Error Bounds
```
python getErr.py --model none --dataset mnist --target-err 0 --stop-err 0.3 --filter-sigma 2 --filter-size 5 --sigma-gamma 30 --nrBetas 1 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_rot_30 --gamma 30 --trans rot

python getErr.py --model none --dataset cifar --target-err 0 --stop-err 0.6 --filter-sigma 1 --filter-size 5 --sigma-gamma 30 --nrBetas 1 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name cifar_rot_30 --gamma 30 --trans rot

python getErr.py --model none --dataset mnist --target-err 0 --stop-err 0.3 --filter-sigma 2 --filter-size 5 --sigma-gamma 2 --nrBetas 1 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_trans_2 --gamma 2 --trans trans --radiusDecrease 4  --initial-splits 10 --refinements 10 --nrBetas 100 --threads 2 --gt-batch-size 20

python eval.py "results/getErr/**"
python getErrSampling.py --dataset imagenet --alpha_min_max 30 --beta_std 30 --trafo rotation --interpolation bilinear --min_size 0 --scale_size 512 --resize-crop-crop 224 --resize-crop-resize 256 --post "Blur(5, 2)" -L 10 --vingette circ -p64 --seed 100 -N 700000
python eval.py "results/getErrSampling/**"
```
 
#### Perform Classification
``` shell
python classifyDistributional.py --dataset mnist --sigmaI 0.30 --sigma-gamma 30 --batch-size 1000 --alphaI 0.002 -E 0.55 --rhoE 0.001 --resize 0 --n-gamma 200 --interpolation bilinear --name mnist_rot30_bilinear --filter-sigma 2.0 --filter-size 5  -N 1000 --model mnist_rot30_sigma03_eps02_60_180_filter5_20
python classifyDistributional.py --dataset mnist --sigmaI 0.30 --sigma-gamma 180 --batch-size 1000 --alphaI 0.002 -E 0.55 --rhoE 0.001 --resize 0 --n-gamma 200 --interpolation bilinear --name mnist_rot180_bilinear --filter-sigma 2.0 --filter-size 5  -N 1000 --model mnist_rot180_sigma03_eps02_60_180_filter5_20
python classifyDistributional.py --transformation rot --interpolation bilinear --dataset cifar --sigmaI 0.25 --sigma-gamma 30 -E 0.77 --rhoE 0.001 --resize 32 --filter-sigma 1.0 --filter-size 5 --n-gamma 50 -nI 10000 --batch-size 2000 --alphaI 0.002 --name cifar_r30 --model cifar_rot30_sigma03_eps1_filter5_10  --resize-post-transform 0 --center-crop-post-transform 0 -N 1000
python classifyDistributional.py --transformation rot --interpolation bilinear --dataset cifar --sigmaI 0.25 --sigma-gamma 30 -E 0.77 --rhoE 0.001 --resize 32 --filter-sigma 1.0 --filter-size 5 --n-gamma 200 -nI 10000 --batch-size 2000 --alphaI 0.002 --name cifar_r30_200 --model cifar_rot30_sigma03_eps1_filter5_10  --resize-post-transform 0 --center-crop-post-transform 0 -N 1000
python classifyDistributional.py --transformation rot --interpolation bilinear --dataset restricted_imagenet --sigmaI 0.5 --sigma-gamma 30 -E 1.20 --rhoE 0.001 --resize 512 --filter-sigma 2.0 --filter-size 5 --n-gamma 50 -nI 2000 --batch-size 200 --alphaI 0.002 --name r_imagenet_rot30_bilinear --model rimagenet_rot30_sigma05_eps1_filter5_20 --resize-post-transform 256 --center-crop-post-transform 224 -N 1000
python classifyDistributional.py --transformation rot --interpolation bilinear --dataset imagenet --sigmaI 0.5 --sigma-gamma 30 -E 1.20 --rhoE 0.001 --resize 512 --filter-sigma 2.0 --filter-size 5 --n-gamma 50 -nI 2000 --batch-size 200 --alphaI 0.002 --name imagenet_rot30_bilinear --model imagenet_rot30_sigma05_eps1_filter5_20 --resize-post-transform 256 --center-crop-post-transform 224 -N 1000
python classifyDistributional.py --dataset mnist --sigmaI 0.30 --sigma-gamma 2 --batch-size 1000 --alphaI 0.002 -E 0.72 --rhoE 0.001 --resize 0 --n-gamma 200 --interpolation bilinear --name mnist_trans4_bilinear --filter-sigma 2.0 --filter-size 5 --transformation trans  --model mnist_trans05_sigma03_eps02_60_180_filter5_20 -N 1000 --gamma 2 --radiusDecrease 4 -nI 1000
python eval.py "results/classifyDistributional/**"
```


#### Verify E
``` shell
python getErr.py --model none --dataset mnist --target-err 0.55 --stop-err 0.55 --filter-sigma 2 --filter-size 5 --sigma-gamma 30 --nrBetas 8000 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_rot_30 --gamma 30 --trans rot
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 0.55 --conf 0.001 --alphaE 0.001 -N 8000
python getErr.py --model none --dataset cifar --target-err 0.77 --stop-err 0.77 --filter-sigma 1 --filter-size 5 --sigma-gamma 30 --nrBetas 8000 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name cifar_rot_30 --gamma 30 --trans rot
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 0.77 --conf 0.001 --alphaE 0.001 -N 8000
python getErrSampling.py --dataset imagenet --alpha_min_max 30 --beta_std 30 --trafo rotation --interpolation bilinear --min_size 0 --scale_size 512 --resize-crop-crop 224 --resize-crop-resize 256 --post "Blur(5, 2)" -L 10 --vingette circ -p64 --seed 100 -N 1000 -K 8000
```

### Individual Defense (IndivSPT, cf. Section 6.3)

We use models from previous section.

``` shell
python classifyOnline.py --dataset mnist --sigmaI 0.4 --sigma-gamma 30 --batch-size 1000 --alphaI 0.002 --resize 0 --n-gamma 2000 --name mnist_rot10_bilinear --filter-sigma 2.0 --filter-size 5  -N 1000 --model mnist_rot30_sigma03_eps02_60_180_filter5_20 --attack-k 100 --nr-attacks 3 --threads 2 --gt-batch-size 20 --stop-err 2.0 --target-err 0.3 --refinements 10 --nrBetas 100 --trans rot --gamma 10 -E 0.7

python classifyOnline.py --dataset mnist --sigmaI 0.3 --sigma-gamma 1.5 --batch-size 1000 --alphaI 0.002 --resize 0 --n-gamma 200 --name mnist_tarns4_bilinear --filter-sigma 2.0 --filter-size 5  -N 100 --model  mnist_trans05_sigma03_eps02_60_180_filter5_20 --attack-k 100 --nr-attacks 3 --threads 2 --gt-batch-size 20 --stop-err 2.0 --target-err 0.35 -E 0.35 --refinements 10 --nrBetas 100 --gamma 1 --trans trans --debug False --initial-splits 10 --radiusDecrease 4

python eval.py "results/classifyOnline/**"
```

## Additional Experiments

### Refinement Experiment (cf. Appendix B.3)
``` shell
python refinementExp.py --model none --dataset mnist --target-err 0 --stop-err 50000000 --nr-attacks 3 --filter-size 5 --filter-sigma 2 -N 20 --name mnist --transformation rot --gamma 10 --sigma-gamma 10  --resize-post-transform 0 --center-crop-post-transform 0 --filter-sigma 2.0 --filter-size 5 --radiusDecrease 2
python eval.py "results/refinementExp/**"
```

### Additional Experiments for Heuristic Defense (BaseSPT, cf. Appendix E.1)
``` shell
python train.py --batch-size 256 --step-lr 30 --epochs 90 --sigmaN 0.0 --eps 0.0 --attack False --vingette none --filter-sigma 0 --filter-size 0 --rotate 0 --out-dir gtsrb_clean_30_90 --crop-resize-resize 32 --crop-resize-crop 32 --lr 0.1 --dataset GTSRB --start-epoch 0

python train.py --batch-size 1024 --attack false --sigmaN 0 --eps 0 --filter-size 0 --filter-sigma 0 --rotate 0 --crop-resize-crop 0 --crop-resize-resize 0 --out-dir fashionmnist_clean_60_180 --dataset fashionmnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0

python classify_f.py --model fashionmnist_clean_60_180 --dataset fashionmnist
python classify_f.py --model gtsrb_clean_30_90 --dataset GTSRB

python classifyHeuristic.py --dataset fashionmnist --model fashionmnist_clean_60_180 -N 1000 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --name heuristic_rot_fashionmnistclean_30_100_1000
python classifyHeuristic.py --dataset fashionmnist --model fashionmnist_clean_60_300 -N 1000 --gamma 4 --sigma-gamma 4 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --trans trans --name heuristic_trans_fashionmnistclean_30_100_1000
python classifyHeuristic.py --dataset GTSRB --model gtsrb_clean_30_90 -N 1000 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --name heuristic_rot_gtsrbclean_30_100_100
python classifyHeuristic.py --dataset GTSRB --model gtsrb_clean_30_90 -N 1000 --gamma 4 --sigma-gamma 4 --attack-k 100 --n0-gamma 100 --n-gamma 1000 --trans trans --name heuristic_trans_gtsrbclean_30_100_1000

python eval.py "results/classifyHeuristic/**"
```

### Radius Experiments for Heuristic Defense (BaseSPT, cf. Appendix E.1)
``` shell
python classifyHeuristicRadius.py --dataset mnist --model mnist_convnet_clean_60_180 -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_mnistclean_30_100_1000
python classifyHeuristicRadius.py --dataset cifar --model cifar_clean_30_90 -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_cifarclean_30_100_100
python classifyHeuristicRadius.py --dataset imagenet --model resnet50 -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_imagenetclean_30_100_1000
python classifyHeuristicRadius.py --dataset fashionmnist --model fashionmnist_clean_60_180 -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_fashionmnistclean_30_100_1000
python classifyHeuristicRadius.py --dataset GTSRB --model gtsrb_clean_30_90 -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_gtsrbclean_30_100_100

python classifyHeuristicRadius.py --dataset mnist --model mnist_convnet_clean_60_300 -N 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --transformation trans --sigma-gamma 4 --gamma 4 --name heuristic_trans_mnistclean_4_100_1000
python classifyHeuristicRadius.py --dataset cifar --model cifar_clean_30_90 -N 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --transformation trans --sigma-gamma 4 --gamma 4 --name heuristic_trans_cifarclean_4_100_1000
python classifyHeuristicRadius.py --dataset imagenet --model resnet50 -N 30 --attack-k 100 --n0-gamma 100 --n-gamma 20 --transformation trans --sigma-gamma 20 --gamma 20 --name heuristic_trans_imagenetclean_20_100_1000
python classifyHeuristicRadius.py --dataset fashionmnist --model fashionmnist_clean_60_300 -N 30 --gamma 4 --sigma-gamma 4 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --trans trans --name heuristic_trans_fashionmnistclean_30_100_1000
python classifyHeuristicRadius.py --dataset GTSRB --model gtsrb_clean_30_90 -N 30 --gamma 4 --sigma-gamma 4 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --trans trans --name heuristic_trans_gtsrbclean_30_100_1000
python eval.py "results/classifyHeuristicRadius/**"
```

``` shell
python train.py --batch-size 1024 --attack false --sigmaN 0 --eps 0 --filter-size 0 --filter-sigma 0 --rotate 0 --crop-resize-crop 28 --crop-resize-resize 28 --out-dir fashionmnist_clean_60_180_v --dataset fashionmnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0 --vingette circ
python train.py --batch-size 256 --step-lr 30 --epochs 90 --sigmaN 0.0 --eps 0.0 --attack False --vingette circ --filter-sigma 0 --filter-size 0 --rotate 0 --out-dir cifar_clean_30_90_v --crop-resize-resize 0 --crop-resize-crop 32 --lr 0.1 --dataset cifar --start-epoch 0
python train.py --batch-size 256 --step-lr 30 --epochs 90 --sigmaN 0.0 --eps 0.0 --attack False --vingette circ --filter-sigma 0 --filter-size 0 --rotate 0 --out-dir gtsrb_clean_30_90_v --crop-resize-resize 32 --crop-resize-crop 32 --lr 0.1 --dataset GTSRB --start-epoch 0

python classifyHeuristicRadius.py --dataset mnist --model mnist_convnet_clean_60_180 -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_mnistclean_30_100_1000  --radiusDecrease 2
python classifyHeuristicRadius.py --dataset fashionmnist --model fashionmnist_clean_60_180_v -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_fashionmnistclean_30_100_1000  --radiusDecrease 2
python classifyHeuristicRadius.py --dataset cifar --model cifar_clean_30_90_v -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_cifarclean_30_100_100_v --radiusDecrease 2
python classifyHeuristicRadius.py --dataset GTSRB --model gtsrb_clean_30_90_v -N 30 --gamma 30 --attack-k 100 --n0-gamma 100 --n-gamma 2000 --name heuristic_rot_gtsrbclean_30_100_100_v --radiusDecrease 2
python eval.py "results/classifyHeuristicRadius/**"
```

### Distributional Defense for Bicubic Error (DistSPT, cf. Appendix E.2)
``` shell
python getErrSampling.py --dataset mnist --alpha_min_max 30 --beta_std 30 --trafo rotation --interpolation bicubic --min_size 0 --scale_size 0 --resize-crop-crop 0 --resize-crop-resize 0 --post "Blur(5, 2)" -L 10 --vingette none -p64 --seed 100 -N 1000
python getErrSampling.py --dataset cifar --alpha_min_max 30 --beta_std 30 --trafo rotation --interpolation bicubic --min_size 0 --scale_size 0 --resize-crop-crop 32 --resize-crop-resize 0 --post "Blur(5, 1)" -L 10 --vingette circ -p64 --seed 100 -N 1000
python getErrSampling.py --dataset imagenet --alpha_min_max 30 --beta_std 30 --trafo rotation --interpolation bicubic --min_size 0 --scale_size 512 --resize-crop-crop 224 --resize-crop-resize 256 --post "Blur(5, 2)" -L 10 --vingette circ -p64 --seed 100 -N 1000
python eval.py "sampling/**"

python classifyDistributional.py --dataset mnist --sigmaI 0.30 --sigma-gamma 30 --batch-size 1000 --alphaI 0.002 -E 0.66 --rhoE 0.001 --resize 0 --n-gamma 200 --interpolation bicubic --name mnist_rot30_bicubic044 --filter-sigma 2.0 --filter-size 5  -N 1000 --model mnist_rot30_sigma03_eps02_60_180_filter5_20
python classifyDistributional.py --transformation rot --interpolation bicubic --dataset cifar --sigmaI 0.25 --sigma-gamma 30 -E 1.11 --rhoE 0.001 --resize 32 --filter-sigma 1.0 --filter-size 5 --n-gamma 50 -nI 10000 --batch-size 2000 --alphaI 0.002 --name cifar_r30_bicubic074 --model cifar_rot30_sigma012_eps025_70_500_filter5_1  --resize-post-transform 0 --center-crop-post-transform 0 -N 100
python classifyDistributional.py --transformation rot --interpolation bicubic --dataset restricted_imagenet --sigmaI 0.75 --sigma-gamma 30 -E 1.84 --rhoE 0.001 --resize 512 --filter-sigma 2.0 --filter-size 5 --n-gamma 50 -nI 4000 --batch-size 200 --alphaI 0.002 --name imagenet_rot30_bicubic152 --model rimagenet_rot30_sigma05_eps1_30_90_v --resize-post-transform 256 --center-crop-post-transform 224 -N 100
python classifyDistributional.py --transformation rot --interpolation bicubic --dataset restricted_imagenet --sigmaI 0.75 --sigma-gamma 30 -E 2.76 --rhoE 0.001 --resize 512 --filter-sigma 2.0 --filter-size 5 --n-gamma 50 -nI 4000 --batch-size 200 --alphaI 0.002 --name imagenet_rot30_bicubic152 --model rimagenet_rot30_sigma05_eps1_30_90_v --resize-post-transform 256 --center-crop-post-transform 224 -N 100
python eval.py "results/classifyDistributional/**"
```

### Audio Volume Change (cf. Appendix E.4)
See [audio/README.md](audio/README.md).

### Ablation Study (cf. Appendix F)

``` shell
python getErr.py --model none --dataset mnist --target-err 0 --stop-err 0.3 --filter-sigma 2 --filter-size 5 --sigma-gamma 30 --nrBetas 1 --radiusDecrease -1 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_rot_30_noV --gamma 30 --trans rot
python getErr.py --model none --dataset mnist --target-err 0 --stop-err 0.3 --filter-sigma 0 --filter-size 1 --sigma-gamma 30 --nrBetas 1 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_rot_30_noG --gamma 30 --trans rot
python getErr.py --model none --dataset mnist --target-err 0 --stop-err 0.3 --filter-sigma 0 --filter-size 1 --sigma-gamma 30 --nrBetas 1 --radiusDecrease 0 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_rot_30_noV_noG --gamma 30 --trans rot

python getErr.py --model none --dataset cifar --target-err 0 --stop-err 0.6 --filter-sigma 1 --filter-size 5 --sigma-gamma 30 --nrBetas 1 --radiusDecrease -1 --initial-splits 1000 --resize 0 -N 1000  --gpu True --threads 2 --name cifar_rot_30_noV --gamma 30 --trans rot
python getErr.py --model none --dataset cifar --target-err 0 --stop-err 0.6 --filter-sigma 0 --filter-size 1 --sigma-gamma 30 --nrBetas 1 --radiusDecrease 2 --initial-splits 1000 --resize 0 -N 1000  --gpu True --threads 2 --name cifar_rot_30_noG --gamma 30 --trans rot
python getErr.py --model none --dataset cifar --target-err 0 --stop-err 0.6 --filter-sigma 0 --filter-size 1 --sigma-gamma 30 --nrBetas 1 --radiusDecrease -1 --initial-splits 1000 --resize 0 -N 1000  --gpu True --threads 2 --name cifar_rot_30_noG_noV --gamma 30 --trans rot

python eval.py "results/getErr/**"
```


### Comparison to DeepG (cf. Appendix F)
To reproduce this experiment setup and use [DeepG](https://github.com/eth-sri/deepg).
In order to add the Gaussian Filter and vignetting replace `deepg/code/abstraction/abstraction.cpp` with the provided [abstraction.cpp](deepg/abstraction.cpp).

## Cite

If you use the code in this repository please cite it as:

```
@incollection{fischer2020transformationsmoothing, title = {Certified Defense to Image Transformations via Randomized Smoothing}, author = { Fischer, Marc and Baader, Maximilian and Vechev, Martin}, booktitle = {Advances in Neural Information Processing Systems 33}, year = {2020} }
```


## Contributes
- [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
- [Maximilian Baader](https://www.sri.inf.ethz.ch/people/max)
- [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)
