import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('-N', type=int, default=3000)
parser.add_argument('-E', type=float, default=0.55)
parser.add_argument('--alphaE', type=float, default=0.1)
parser.add_argument('--rhoE', type=float, default=0.001)
args = parser.parse_args()

df = pd.read_csv(args.path, skiprows=2, sep=' ', names=['index', 'err'])
df['err'] = df['err'] <= args.E
cnts = df.groupby(['index']).sum().to_numpy()[:, 0]
f = lambda x: 1-proportion_confint(x, args.N, alpha=2*args.alphaE, method="beta")[0]
aEs = np.array(list((map(f, cnts))))
cnt = (aEs <= args.rhoE).sum() - args.alphaE * aEs.shape[0]


print(proportion_confint(cnt, aEs.shape[0], alpha=2*args.alphaE, method="beta")[0])
