This code is based on [GCommandsPytorch](https://github.com/adiyoss/GCommandsPytorch).


### Install

Download and extract the [Google CommandsDataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html) to the `GCommandsPytorch` folder and then:

``` shell
cd GCommandsPytorch
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
tar xvzf speech_commands_v0.01.tar.gz
python make_dataset.py . --out_path data
cd -
```

### Training Classifier

The model can also be downloaded [here](https://files.sri.inf.ethz.ch/transformation-smoothing/model_audio.tar.gz).

``` shell
python train.py --lr 0.01 --trafo volume --param 10 --sigma 0.006  --out-dir "gcommands_volume10_sigma0006"
```


### Estimating E

```
python sample_error.py -K 10 --alpha_min_max 3 --beta_std 3
python
>>> import pandas as pd
>>> df = pd.read_csv('sampling/volume_51088_10_3.0_3.0_0.csv', names=["idx", "beta", "gamma", "l1", "l2", "linf"] + ["other"] * 6)
>>> (df.groupby('idx').max()["l2"] <= 0.005).sum()/51088
```


### Evaluating Robust Classifiers

``` shell
python distributional.py -N 100 --model gcommands_volume10_sigma0006 -nI 400 --sigmaI 0.006 --n-gamma 150 --sigma-gamma 3 -E 0.005 --rhoE 0.05 --name gcommands_volume3 --batch-size 200
python ../eval.py "results/**"
```

### Individual Online Results


``` shell
python inverse_online_error.py -N 100
awk '{if ($1 <= 0.005) cnt+=1} END{print cnt}' indiv_err_bound.txt
awk '{if ($1 <= 0.0055) cnt+=1} END{print cnt}' indiv_err_bound.txt
```
