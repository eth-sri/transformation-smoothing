Certified Defense to Image Transformations via Randomized Smoothing <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
=============================================================================================================

We present extend an extension of randomized smoothing to cover
parametrized transformations (here image rotations and translations)
and certify robustness in the parameter space (e.g., rotation angle).
Here we evaluate three defense with different guarantees based
this. For further details, please see [our NeurIPS 2020
paper](https://www.sri.inf.ethz.ch/publications/fischer2020smoothing).

![img](https://raw.githubusercontent.com/eth-sri/transformation-smoothing/main/img.png)



## Note 
Since the publication at NeurIPS'20 we found and fixed a bug. This changed some results.
Further, we also added additional experiments to allow better comparison with related approaches.
Check out the latest version of the paper for these results.

We fixed a further bug in [indivSPT_certify_attacked.py](indivSPT_certify_attacked.py) identified by [Maurice Weber](https://systems.ethz.ch/people/profile.MTgxODU3.TGlzdC8zODg4LDEyOTU2NDI2OTI=.html). Thank you, Maurice! This did not change any results.



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
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
cd geometrictools
make python # installs python bindings for our C++ library; needs to be executed in the correct python environemnt for details see geometrictools/README.md
cd -
```

Make sure that your CUDA version (seen in the top right of `nvidia-smi`) is at least 10.2, else replace the cudatoolkit version above with your version.

The folder `smoothing-adversarial` contains a modified version of [smothing-adversarial](https://github.com/Hadisalman/smoothing-adversarial).

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
python train.py --batch-size 256 --sigmaN 0.22 --eps 0.2 --filter-size 5 --filter-sigma 2.0 --rotate 90 --crop-resize-crop 0 --crop-resize-resize 0 --out-dir mnist_rot90_sigma022_eps02_60_180_filter5_20 --dataset mnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0 --crop-resize-resize 0 --crop-resize-crop 28

python train.py --batch-size 256 --sigmaN 0.15 --eps 0.2 --filter-size 5 --filter-sigma 2.0 --rotate 90 --crop-resize-crop 0 --crop-resize-resize 0 --out-dir mnist_rot90_sigma015_eps02_60_180_filter5_20 --dataset mnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0 --crop-resize-resize 0 --crop-resize-crop 28

python train.py --batch-size 256 --sigmaN 0.3 --eps 0.2 --filter-size 5 --filter-sigma 2.0 --translate 0.5 --rotate 0 --crop-resize-crop 0 --crop-resize-resize 0 --out-dir mnist_trans05_sigma03_eps02_60_180_filter5_20 --dataset mnist --lr 0.01 --step-lr 60 --epochs 180 --start-epoch 0 --crop-resize-resize 0 --crop-resize-crop 28

cd smoothing-adversarial
python code/train_pgd.py cifar10 cifar_resnet110 cifar_rot60_s25_e05_t8_m1 --batch 256 --noise 0.25 --noise_sd 0.25 --gpu 0 --lr_step_size 50 --epochs 150 --adv-training --attack PGD --num-steps 8 --epsilon 128 --train-multi-noise --num-noise-vec 1 --warmup 10 --rot 60
cp -r cifar_rot60_s25_e05_t8_m1  ../models 
python code/train_pgd.py cifar10 cifar_resnet110 cifar_rot60_s12_e025_t1_m8 --batch 256 --noise 0.12 --noise_sd 0.12 --gpu 0 --lr_step_size 50 --epochs 150 --adv-training --attack PGD --num-steps 1 --epsilon 64 --train-multi-noise --num-noise-vec 8 --warmup 10 --rot 60
cp -r cifar_rot60_s12_e05_t81_m8  ../models 
IMAGENET_DIR=~/imagenet taskset -c 0-11,24-35 python code/train_pgd.py imagenet resnet50 imagenet_rot30_se05_e05 --batch 256 --noise 0.50 --noise_sd 0.5 --gpu 0,1,2,3,4,5 --lr_step_size 30 --epochs 90 --adv-training --attack PGD --num-steps 1 --epsilon 128 --num-noise-vec 1 --warmup 10 --batch 400 --workers 24 --resume --rot 30
cp -r imagenet_rot30_se05_e05 ../models
IMAGENET_DIR=~/imagenet python code/train_pgd.py restricted_imagenet resnet50 restricted_imagenet_rot60_se05_e05 --batch 256 --noise 0.50 --noise_sd 0.5 --gpu 0,1,2,3,4,5 --lr_step_size 30 --epochs 90 --adv-training --attack PGD --num-steps 1 --epsilon 128 --num-noise-vec 1 --warmup 10 --batch 400 --workers 24 --resume --rot 60
cp -r restricted_imagenet_rot60_se05_e05  ../models
cd ..

python classify_f.py --dataset mnist --model mnist_rot90_sigma022_eps02_60_180_filter5_20 
python classify_f.py --dataset mnist --model mnist_rot90_sigma022_eps02_60_180_filter5_20 --filter-sigma 2.0 --filter-size 5
python classify_f.py --dataset mnist --model mnist_rot90_sigma022_eps02_60_180_filter5_20 --radiusDecrease 2
python classify_f.py --dataset mnist --model mnist_rot90_sigma022_eps02_60_180_filter5_20 --radiusDecrease 2 --filter-sigma 2.0 --filter-size 5

python classify_f.py --dataset mnist --model mnist_rot90_sigma015_eps02_60_180_filter5_20 
python classify_f.py --dataset mnist --model mnist_rot90_sigma015_eps02_60_180_filter5_20 --filter-sigma 2.0 --filter-size 5
python classify_f.py --dataset mnist --model mnist_rot90_sigma015_eps02_60_180_filter5_20 --radiusDecrease 2
python classify_f.py --dataset mnist --model mnist_rot90_sigma015_eps02_60_180_filter5_20 --radiusDecrease 2 --filter-sigma 2.0 --filter-size 5

python classify_f.py --dataset mnist --model mnist_trans05_sigma03_eps02_60_180_filter5_20 
python classify_f.py --dataset mnist --model mnist_trans05_sigma03_eps02_60_180_filter5_20 --filter-sigma 2.0 --filter-size 5
python classify_f.py --dataset mnist --model mnist_trans05_sigma03_eps02_60_180_filter5_20 --radiusDecrease 4 --trans trans 
python classify_f.py --dataset mnist --model mnist_trans05_sigma03_eps02_60_180_filter5_20 --radiusDecrease 4 --trans trans --filter-sigma 2.0 --filter-size 5

python classify_f.py --dataset cifar --model cifar_rot60_s25_e05_t8_m1 
python classify_f.py --dataset cifar --model cifar_rot60_s25_e05_t8_m1 --filter-sigma 1.0 --filter-size 5
python classify_f.py --dataset cifar --model cifar_rot60_s25_e05_t8_m1 --radiusDecrease 2
python classify_f.py --dataset cifar --model cifar_rot60_s25_e05_t8_m1 --radiusDecrease 2 --filter-sigma 1.0 --filter-size 5

python classify_f.py --dataset cifar --model cifar_rot60_s12_e025_t1_m8 
python classify_f.py --dataset cifar --model cifar_rot60_s12_e025_t1_m8 --filter-sigma 1.0 --filter-size 5
python classify_f.py --dataset cifar --model cifar_rot60_s12_e025_t1_m8 --radiusDecrease 2
python classify_f.py --dataset cifar --model cifar_rot60_s12_e025_t1_m8 --radiusDecrease 2 --filter-sigma 1.0 --filter-size 5

python classify_f.py --dataset imagenet --batch-size 200 --model imagenet_rot30_se05_e05 --resize-post-transform 256 --center-crop-post-transform 224
python classify_f.py --dataset imagenet --batch-size 200 --model imagenet_rot30_se05_e05 --resize-post-transform 256 --center-crop-post-transform 224 --filter-sigma 2.0 --filter-size 5
python classify_f.py --dataset imagenet --batch-size 200 --model imagenet_rot30_se05_e05 --resize-post-transform 256 --center-crop-post-transform 224 --radiusDecrease 2
python classify_f.py --dataset imagenet --batch-size 200 --model imagenet_rot30_se05_e05 --resize-post-transform 256 --center-crop-post-transform 224 --radiusDecrease 2 --filter-sigma 2.0 --filter-size 5

python classify_f.py --dataset restricted_imagenet --batch-size 200 --model restricted_imagenet_rot60_se05_e05 --resize-post-transform 256 --center-crop-post-transform 224
python classify_f.py --dataset restricted_imagenet --batch-size 200 --model restricted_imagenet_rot60_se05_e05 --resize-post-transform 256 --center-crop-post-transform 224 --filter-sigma 2.0 --filter-size 5
python classify_f.py --dataset restricted_imagenet --batch-size 200 --model restricted_imagenet_rot60_se05_e05  --resize-post-transform 256 --center-crop-post-transform 224 --radiusDecrease 2
python classify_f.py --dataset restricted_imagenet --batch-size 200 --model restricted_imagenet_rot60_se05_e05  --resize-post-transform 256 --center-crop-post-transform 224 --radiusDecrease 2 --filter-sigma 2.0 --filter-size 5
```

#### Obtain and verify E 
```
python getErr.py --model none --dataset mnist --target-err 0 --stop-err 0.3 --filter-sigma 2 --filter-size 5 --sigma-gamma 30 --nrBetas 1 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_rot_90 --gamma 90 --trans rot

python getErr.py --model none --dataset cifar --target-err 0 --stop-err 0.6 --filter-sigma 1 --filter-size 5 --sigma-gamma 30 --nrBetas 1 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name cifar_rot_30 --gamma 30 --trans rot

python getErr.py --model none --dataset mnist --target-err 0 --stop-err 0.55 --filter-sigma 2 --filter-size 5 --sigma-gamma 2 --nrBetas 1 --radiusDecrease 4 --trans trans --initial-splits 50 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_trans_2 --gamma 2 --refinements 10 --gt-batch-size 20


python getErr.py --model none --dataset mnist --filter-sigma 2 --filter-size 5 --sigma-gamma 2 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_trans_2_1000_8000 --gamma 2 --trans trans --radiusDecrease 4 --initial-splits 50 --refinements 10 --nrBetas 8000 --threads 2 --target-err 0.55 --stop-err 0.55 --nrBetasSplit 8000

python eval.py "results/getErr/**"

python getErr.py --model none --dataset mnist --target-err 0.45 --stop-err 0.45 --filter-sigma 2 --filter-size 5 --sigma-gamma 30 --nrBetas 8000 --radiusDecrease 2 --initial-splits 1000 --resize 0 -N 1000 --gpu True --threads 2 --name mnist_rot_90 --gamma 90 --trans rot --betaSplit 1000
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 0.45 --rhoE 0.001 --alphaE 0.001 -N 8000


python getErr.py --model none --dataset cifar --target-err 0.55 --stop-err 0.55 --filter-sigma 1 --filter-size 5 --sigma-gamma 30 --nrBetas 1 --radiusDecrease 2 --initial-splits 100 --resize 0 -N 1000  --gpu True --threads 2 --name cifar_rot_30 --gamma 30 --trans rot  --betaSplit 1000
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 0.55 --rhoE 0.001 --alphaE 0.001 -N 8000

python getErr.py --model none --dataset mnist --target-err 0.65 --stop-err 0.65 --filter-sigma 2 --filter-size 5 --sigma-gamma 2 --nrBetas 8000 --radiusDecrease 4 --trans trans --initial-splits 50 --resize 0 -N 1000  --gpu True --threads 2 --name mnist_trans_2 --gamma 2 --refinements 10 --gt-batch-size 20  --betaSplit 1000
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 0.65 --rhoE 0.001 --alphaE 0.001 -N 8000

python getErrSampling.py --transformation rot --interpolation bilinear --dataset imagenet --model none -N 1000 --sigma-gamma 30 --gamma 30 --resize 512 --filter-sigma 2.0 --filter-size 5 --resize-post-transform 256 --center-crop-post-transform 224 --nrBetas 1 --nrSamples 10 --threads 128 --radiusDecrease 2 --name imangenet_1 

# The following lbock should be run for E 0.95, 1.15, 1.20, 1.35
# To avoid multiple testing bias the sampling should also be rerun for each E; to this end pass --seed with a random number to it
# Also adjust --threads to the number of availiable cores
python getErrSampling.py --transformation rot --interpolation bilinear --dataset imagenet --model none -N 1000 --sigma-gamma 30 --gamma 30 --resize 512 --filter-sigma 2.0 --filter-size 5 --resize-post-transform 256 --center-crop-post-transform 224 --nrBetas 8000 --nrSamples 10 --threads 128 --radiusDecrease 2 --name imangenet_8000
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 1.35 --rhoE 0.001 --alphaE 0.001 -N 8000
```

#### Perform Certification for distSPTD 
``` shell

python distSPTD_certify.py  --transformation rot --interpolation bilinear --dataset mnist -N 1000 --resize 0 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 10000 -E 0.45 --rhoE 0.001  --sigma-eps 0.25 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 30  --alpha-gamma 0.004 --model mnist_rot90_sigma022_eps02_60_180_filter5_20 --name  mnist_rot_bilinear

python distSPTD_certify.py  --transformation trans --interpolation bilinear --dataset mnist -N 100 --resize 0 --radiusDecrease 4 --filter-sigma 2 --filter-size 5 --batch-size 10000 -E 0.65 --rhoE 0.001  --sigma-eps 0.25 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 1.5  --alpha-gamma 0.004 --model mnist_trans05_sigma03_eps02_60_180_filter5_20  --name  mnist_trans_bilinear

python distSPTD_certify.py --transformation rot --interpolation bilinear --dataset cifar -N 1000 --resize 0 --radiusDecrease 2 --filter-sigma 1 --filter-size 5 --batch-size 2000 -E 0.55 --rhoE 0.001  --sigma-eps 0.3 --alpha-eps 0.005 --n-eps 15000 --n-gamma 50 --n0-gamma 10000 --sigma-gamma 30  --alpha-gamma 0.004 --model cifar_rot60_s25_e05_t8_m1 --name  cifar_rot_bilinear_n50

python distSPTD_certify.py --transformation rot --interpolation bilinear --dataset cifar -N 1000 --resize 0 --radiusDecrease 2 --filter-sigma 1 --filter-size 5 --batch-size 2000 -E 0.55 --rhoE 0.001  --sigma-eps 0.3 --alpha-eps 0.005 --n-eps 15000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 30  --alpha-gamma 0.004 --model cifar_rot60_s25_e05_t8_m1 --name  cifar_rot_bilinear_n200

python distSPTD_certify.py --transformation rot --interpolation bilinear --dataset restricted_imagenet -N 1000 --resize 512 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 200 -E 1.20 --rhoE 0.001  --sigma-eps 0.5 --alpha-eps 0.005 --n-eps 2500 --n-gamma 50 --n0-gamma 200 --sigma-gamma 30 --alpha-gamma 0.004 --resize-post-transform 256  --center-crop-post-transform 224 --model restricted_imagenet_rot60_se05_e05 --name rimagenet_rot_bilinear_E120

python distSPTD_certify.py --transformation rot --interpolation bilinear --dataset restricted_imagenet -N 1000 --resize 512 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 200 -E 1.35 --rhoE 0.001  --sigma-eps 0.55 --alpha-eps 0.005 --n-eps 2500 --n-gamma 50 --n0-gamma 200 --sigma-gamma 30 --alpha-gamma 0.004 --resize-post-transform 256  --center-crop-post-transform 224 --model restricted_imagenet_rot60_se05_e05 --name rimagenet_rot_bilinear_E135

python distSPTD_certify.py --transformation rot --interpolation bilinear --dataset imagenet -N 1000 --resize 512 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 200 -E 1.15 --rhoE 0.001  --sigma-eps 0.5 --alpha-eps 0.005 --n-eps 2500 --n-gamma 50 --n0-gamma 200 --sigma-gamma 30 --alpha-gamma 0.004 --resize-post-transform 256  --center-crop-post-transform 224 --model imagenet_rot30_se05_e05 --name imagenet_rot_bilinear_E115

python eval.py "results/distSPTD_certify/**"

python distSPTD_certify_attacked.py  --transformation rot --interpolation bilinear --dataset mnist -N 1000 --resize 0 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 10000 -E 0.45 --rhoE 0.001  --sigma-eps 0.25 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 30  --alpha-gamma 0.004 --model mnist_rot90_sigma022_eps02_60_180_filter5_20 --name  mnist_rot_bilinear --nr-attacks 1 --attack-k 100 --gamma 30 

python eval.py "results/distSPTD_certify_attacked/**"
```

#### Perform Certification for distSPTx
``` shell
python distSPTx_certify.py --transformation rot --interpolation bilinear --dataset mnist -N 1000 --resize 0 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 10000 --gamma 50 --nrBetas 500 --target-err 0.3 --stop-err 2.0 --gt-batch-size 20 --threads 2  --alpha-E 0.001 --guess-E-samples 100 --guess-E-mult 1.1 --guess-E-add 0.0 --sigma-eps 0.25 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 30  --alpha-gamma 0.004 --model mnist_rot90_sigma015_eps02_60_180_filter5_20 --name  mnist_rot

python distSPTx_certify.py --transformation trans --interpolation bilinear --dataset mnist -N 100 --resize 0 --radiusDecrease 4 --filter-sigma 2 --filter-size 5 --batch-size 10000 --gamma 2 --nrBetas 500 --target-err 0.55 --stop-err 2.0 --gt-batch-size 20 --threads 2  --alpha-E 0.001 --guess-E-samples 100 --guess-E-mult 1.0 --guess-E-add 0.02 --sigma-eps 0.25 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 2.5 --alpha-gamma 0.004 --model mnist_trans05_sigma03_eps02_60_180_filter5_20 --name  mnist_trans

python distSPTx_certify.py --transformation rot --interpolation bilinear --dataset cifar -N 1000 --resize 0 --radiusDecrease 2 --filter-sigma 1 --filter-size 5 --batch-size 2000 --gamma 30 --nrBetas 500 --target-err 0.3 --stop-err 2.0 --gt-batch-size 20 --threads 2  --alpha-E 0.001 --guess-E-samples 100 --guess-E-mult 1.1 --guess-E-add 0.0 --sigma-eps 0.2 --alpha-eps 0.005 --n-eps 15000 --n-gamma 50 --n0-gamma 10000 --sigma-gamma 40  --alpha-gamma 0.004 --model cifar_rot60_s12_e025_t1_m8 --name  cifar_rot_30

python distSPTx_certify.py --transformation rot --interpolation bilinear --dataset cifar -N 1000 --resize 0 --radiusDecrease 2 --filter-sigma 1 --filter-size 5 --batch-size 2000 --gamma 10 --nrBetas 500 --target-err 0.3 --stop-err 2.0 --gt-batch-size 20 --threads 2 --alpha-E 0.001 --guess-E-samples 100 --guess-E-mult 1.1 --guess-E-add 0.0 --sigma-eps 0.2 --alpha-eps 0.005 --n-eps 15000 --n-gamma 50 --n0-gamma 10000 --sigma-gamma 10 --alpha-gamma 0.004 --model cifar_rot60_s12_e025_t1_m8 --name  cifar_rot_10

# The next two calls can be expensive due to the sampling. To this end it may be helpful to reuse the E values computed for 1 pass for the second. This can be done with --Efile /path/to result of first call. 
# In particular it may be useful to just precompute the E values on a machine with many CPU cores. To this end use --model none (to skip the actual certication) and set --threads to the number of availiable CPU cores.
python distSPTx_certify.py --transformation rot --interpolation bilinear --dataset imagenet -N 1000 --resize 512 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 200 --gamma 30 --nrBetas 640 --threads 16 --nrSamples 10 --sampleE True --alpha-E 0.001 --guess-E-samples 240 --guess-E-mult 1.0 --guess-E-add 0.003 --sigma-eps 0.5 --alpha-eps 0.005 --n-eps 2500 --n-gamma 50 --n0-gamma 50 --sigma-gamma 30 --alpha-gamma 0.004 --resize-post-transform 256  --center-crop-post-transform 224 --model imagenet_rot30_se05_e05  --name imagenet_rot_n50

python distSPTx_certify.py --transformation rot --interpolation bilinear --dataset imagenet -N 1000 --resize 512 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 200 --gamma 30 --nrBetas 640 --threads 16 --nrSamples 10 --sampleE True --alpha-E 0.001 --guess-E-samples 240 --guess-E-mult 1.0 --guess-E-add 0.003 --sigma-eps 0.5 --alpha-eps 0.005 --n-eps 2500 --n-gamma 50 --n0-gamma 200 --sigma-gamma 30 --alpha-gamma 0.004 --resize-post-transform 256  --center-crop-post-transform 224 --model imagenet_rot30_se05_e05  --name imagenet_rot_n200

python eval.py "results/distSPTx_certify/**"
```


### Individual Defense (IndivSPT, cf. Section 6.3)

We use models from previous section.

``` shell

python indivSPT_certify_attacked.py --transformation rot --interpolation bilinear --dataset mnist -N 1000 --resize 0 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 10000 --nr-attacks 3 --attack-k 100 -E 0.45  --nrBetas 500 --gamma 10 --refinements 10 --target-err 0.3 --stop-err 2.0 --gt-batch-size 20 --threads 2 --sigma-eps 0.3 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 30  --alpha-gamma 0.004 --model mnist_rot90_sigma022_eps02_60_180_filter5_20 --name  mnist_rot

python indivSPT_certify_attacked.py --transformation trans --interpolation bilinear --dataset mnist -N 100 --resize 0 --radiusDecrease 4 --filter-sigma 2 --filter-size 5 --batch-size 10000 --nr-attacks 3 --attack-k 100 -E 0.45 --nrBetas 500 --gamma 1 --refinements 10 --target-err 0.3 --stop-err 2.0 --gt-batch-size 20 --threads 2 --sigma-eps 0.3 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 1.5 --alpha-gamma 0.004 --model  mnist_trans05_sigma03_eps02_60_180_filter5_20 --name  mnist_trans

python eval.py "results/indivSPT_certify_attacked/**"
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
#see above hits on multiple testing and long run times

python getErrSampling.py --transformation rot --interpolation bicubic --dataset mnist --model none -N 1000 --sigma-gamma 30 --gamma 30 --resize 0 --filter-sigma 2.0 --filter-size 5 --nrBetas 1 --nrSamples 10 --threads 128 --radiusDecrease 2 --name mnist_1_bicubic
python eval.py "results/getErrSampling/**"

python getErrSampling.py --transformation rot --interpolation bicubic --dataset mnist --model none -N 1000 --sigma-gamma 30 --gamma 30 --resize 0 --filter-sigma 2.0 --filter-size 5 --nrBetas 8000 --nrSamples 10 --threads 128 --radiusDecrease 2 --name mnist_8000_bicubic
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 0.5 --rhoE 0.001 --alphaE 0.001 -N 8000

python getErrSampling.py --transformation rot --interpolation bicubic --dataset cifar --model none -N 1000 --sigma-gamma 30 --gamma 30 --resize 0 --filter-sigma 1.0 --filter-size 5 --nrBetas 1 --nrSamples 10 --threads 128 --radiusDecrease 2 --name cifar_1_bicubic
python eval.py "results/getErrSampling/**"

python getErrSampling.py --transformation rot --interpolation bicubic --dataset cifar --model none -N 1000 --sigma-gamma 30 --gamma 30 --resize 0 --filter-sigma 1.0 --filter-size 5 --nrBetas 8000 --nrSamples 10 --threads 128 --radiusDecrease 2 --name cifar_8000_bicubic
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 1.10 --rhoE 0.001 --alphaE 0.001 -N 8000

python getErrSampling.py --transformation rot --interpolation bicubic --dataset imagenet --model none -N 1000 --sigma-gamma 30 --gamma 30 --resize 512 --filter-sigma 2.0 --filter-size 5 --resize-post-transform 256 --center-crop-post-transform 224 --nrBetas 1 --nrSamples 10 --threads 128 --radiusDecrease 2 --name imangenet_1_bicubic
python eval.py "results/getErrSampling/**"

python getErrSampling.py --transformation rot --interpolation bicubic --dataset imagenet --model none -N 1000 --sigma-gamma 30 --gamma 30 --resize 512 --filter-sigma 2.0 --filter-size 5 --resize-post-transform 256 --center-crop-post-transform 224 --nrBetas 8000 --nrSamples 10 --threads 128 --radiusDecrease 2 --name imangenet_8000_bicubic
python verifyE.py PATH_TO_RESULT_FROM_PREVIOUS_GET_ERR_FILE -E 2.50 --rhoE 0.001 --alphaE 0.001 -N 8000

python distSPTD_certify.py  --transformation rot --interpolation bicubic --dataset mnist -N 100 --resize 0 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 10000 -E 0.5 --rhoE 0.001  --sigma-eps 0.25 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 30  --alpha-gamma 0.004 --model mnist_rot90_sigma022_eps02_60_180_filter5_20 --name  mnist_rot_bicubic

python distSPTD_certify.py  --transformation rot --interpolation bilinear --dataset mnist -N 100 --resize 0 --radiusDecrease 2 --filter-sigma 2 --filter-size 5 --batch-size 10000 -E 0.45 --rhoE 0.001  --sigma-eps 0.25 --alpha-eps 0.005 --n-eps 10000 --n-gamma 200 --n0-gamma 10000 --sigma-gamma 30  --alpha-gamma 0.004 --model mnist_rot90_sigma022_eps02_60_180_filter5_20 --name  mnist_rot_bilinear_100

python eval.py "results/distSPTD_certify/**"
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



