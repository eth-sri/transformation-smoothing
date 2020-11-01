import pandas as pd
from glob2 import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import argparse
import traceback
from statsmodels.stats.proportion import proportion_confint

def parseClassifyHeuristic(path):
    df = pd.read_csv(path, skiprows=1, sep=' ')
    n = df.shape[0]
    print('n=', n)
    print('f acc', (df['label'] == df['predicted']).sum() / n)
    print('f acc, attacked', (df['label'] == df['label_attack']).sum() / n)
    print('g acc, attacked', ((df['label'] == df['label_smooth']) & (df['radius'] > 0) ).sum() / n)
    print('g Radius q=0.25', df['radius'].quantile(q=0.25))
    print('g Radius q=0.50', df['radius'].quantile(q=0.50))
    print('g Radius q=0.75', df['radius'].quantile(q=0.75))
    print('g time', df['time'].mean())

def parseClassifyHeuristicRadius(path):
    with open(path, 'r') as f:
        settings = f.readline()
    print(settings)
    df = pd.read_csv(path, skiprows=1, sep=' ')
    n = df.shape[0]
    k = 100
    print('radius median', df['radius'].quantile(q=0.5))
    print('incorrect rate', df['incorrect'].sum() / (n*k))
    print('incorrect images', (df['incorrect'] > 0 ).sum() / n)
    
def parseClassifyDistributional(path):
    df = pd.read_csv(path, skiprows=1, sep=' ')
    n = df.shape[0]
    print('f acc', (df['label'] == df['predicted']).sum() / n)
    df = df[df['radius'] != 'None']
    df['radius'] = pd.to_numeric(df['radius'], errors='coerce')
    df['label_smooth'] = pd.to_numeric(df['label_smooth'], errors='coerce')
    I_corr = df['label'] == df['label_smooth']
    df_corr = df[I_corr]
    print('g acc', (df_corr['radius'] >= 0).sum() / n)
    print('g acc @ 0.006', (df_corr['radius'] >= 0.006).sum() / n)
    print('g acc @ 1', (df_corr['radius'] >= 1).sum() / n)
    print('g acc @ 1.41', (df_corr['radius'] >= 1).sum() / n)
    print('g acc @ 2', (df_corr['radius'] >= 2).sum() / n)
    print('g acc @ 3', (df_corr['radius'] >= 3).sum() / n)
    print('g acc @ 4', (df_corr['radius'] >= 4).sum() / n)
    print('g acc @ 5', (df_corr['radius'] >= 5).sum() / n)
    print('g acc @ 10', (df_corr['radius'] >= 10).sum() / n)
    print('g acc @ 20', (df_corr['radius'] >= 20).sum() / n)
    print('g acc @ 30', (df_corr['radius'] >= 30).sum() / n)
    print('g acc @ 50', (df_corr['radius'] >= 50).sum() / n)
    print('g Radius q=0.25', df_corr['radius'].quantile(q=0.25))
    print('g Radius q=0.50', df_corr['radius'].quantile(q=0.50))
    print('g Radius q=0.75', df_corr['radius'].quantile(q=0.75))
    print('g time', pd.to_numeric(df_corr['time']).mean())


def parseClassifyDistributionalJustPrediction(path):
    df = pd.read_csv(path, skiprows=2, sep=' ', names=["index", "label", "predicted", "label_smooth", "radius", "E", "rho", "time", "tE"])
    n = df.shape[0]
    print('avg E', df['E'].sum()/n)
    print('f acc', (df['label'] == df['predicted']).sum() / n)
    df = df[df['radius'] != 'None']
    df['radius'] = pd.to_numeric(df['radius'], errors='coerce')
    df['label_smooth'] = pd.to_numeric(df['label_smooth'], errors='coerce')
    I_corr = df['label'] == df['label_smooth']
    df_corr = df[I_corr]
    print('g acc', (df_corr['radius'] >= 0).sum() / n)
    print('g acc @ 1', (df_corr['radius'] >= 1).sum() / n)
    print('g acc @ 1.41', (df_corr['radius'] >= 1.41).sum() / n)
    print('g acc @ 2', (df_corr['radius'] >= 2).sum() / n)
    print('g acc @ 4', (df_corr['radius'] >= 4).sum() / n)
    print('g acc @ 5', (df_corr['radius'] >= 5).sum() / n)
    print('g acc @ 10', (df_corr['radius'] >= 10).sum() / n)
    print('g acc @ 20', (df_corr['radius'] >= 20).sum() / n)
    print('g acc @ 30', (df_corr['radius'] >= 30).sum() / n)
    print('g acc @ 50', (df_corr['radius'] >= 50).sum() / n)
    print('g Radius q=0.25', df_corr['radius'].quantile(q=0.25))
    print('g Radius q=0.50', df_corr['radius'].quantile(q=0.50))
    print('g Radius q=0.75', df_corr['radius'].quantile(q=0.75))
    print('g time', pd.to_numeric(df_corr['time']).mean())
    print('g time', pd.to_numeric(df['tE']).mean())

    
def parseClassifyOnline(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        header, lines = lines[1], lines[2:]
        l = lines[0]
        import re
        from io import StringIO
        lines_out = [header]
        for l in lines:
            l = re.sub('(\[.*(?:\d|\.))(\s+)(\d.*\])', r'\1,\3', l, count=10)
            l = re.sub('(\[.*(?:\d|\.))(\s*)(\])', r'\1\3', l)
            lines_out.append(l)
        f = StringIO("".join(lines_out))
        df = pd.read_csv(f, skiprows=0, sep=' ')

        print('maxErr', df['maxErr'].max())
        n = df.shape[0]
        # online
        I = df['label'] == df['predicted']
        print('f acc', (I.sum() / n))
        dfI = df.loc[I]
        #print(dfI)
        I = (df['label'] == df['label_attack'])
        I = dfI['label_attack'].astype(int) == dfI['label']
        #print(I)
        print('f acc, attacked', I.sum() / n)
        dfI = dfI.loc[I]
        I = dfI['rhoE'].astype(float) < 0.5
        print('inv successful ', I.sum() / n)
        dfI = dfI.loc[I]
        I = dfI['radius'] != 'None'
        dfI = dfI.loc[I]
        dfI['radius'] = pd.to_numeric(dfI['radius'], errors='coerce')
        dfIacc = dfI[dfI['label'] == dfI['label_smooth']]
        print('g acc', (dfI['radius'] >= 0).sum() / n)
        print('g acc @ 1', (dfI['radius'] >= 1).sum() / n)
        print('g acc @ 1.41', (dfI['radius'] >= 1.41).sum() / n)
        print('g acc @ 10', (dfI['radius'] >= 10).sum() / n)
        R = dfI['radius'].astype(float)
        print('g Radius q=0.25', R.quantile(q=0.25))
        print('g Radius q=0.50', R.quantile(q=0.50))
        print('g Radius q=0.75', R.quantile(q=0.75))
        t0 = dfI['time_analysis'].astype(float).mean()
        t1 = dfI['time_smooth'].astype(float).mean()
        print('g time analsysis', t0)
        print('g time smooth', t1)
        print('g time total', t0+t1)
    
    
def parseGetErr(path):
    with open(path, 'r') as f:
        settings = f.readline()
    df = pd.read_csv(path, skiprows=1, sep=' ')
    print(settings)
    print(df['err'].max())

def parseRefinementExp(path):
    df = pd.read_csv(path, skiprows=0, sep=' ', names=['idx', 'nr. attack', 'refinements', 'err', 'runtime'])
    dfA = df.pivot_table(index='refinements',  values=['err', 'runtime'])
    err = np.array(dfA['err'])
    x = np.array(range(len(dfA.index)))
    fig = plt.figure()
    plt.bar(x, err)
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(dfA.index)
    plt.xlabel('# refinement steps')
    plt.ylabel('E', rotation=0)        
    ax.set_facecolor( (0.97, 0.97, 0.97) )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_coords(0.01, 1.02)
    for i, e in enumerate(err.tolist()):            
        ax.text(i - 0.35, e - .04, f"{e:.2f}", color='white', fontweight='bold', multialignment='center')        
    fig.tight_layout()
    plt.savefig(path.replace('.txt', '_err.png'))

    times = np.array(dfA['runtime'])        
    fig = plt.figure()
    plt.bar(x, times)
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(dfA.index)
    plt.xlabel('# refinement steps')
    plt.ylabel('run time [s]', rotation=0)        
    ax.set_facecolor( (0.97, 0.97, 0.97) )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_coords(0.01, 1.02)
    for i, t in enumerate(times.tolist()):
        d = (int(t) >= 10)
        ax.text(i - (0.25 + d * 0.1) , t - 1.5, f"{t:.1f}", color='white', fontweight='bold', multialignment='center')        
    fig.tight_layout()
    plt.savefig(path.replace('.txt', '_times.png'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    for path in glob(args.path):
        print()
        print('*' * 20)
        path = os.path.normpath(path)
        tokens = path.split(os.sep)
        if (tokens[0].startswith('results') and len(tokens) == 3):
            print(path)
            try:
                if tokens[0].startswith('results'):
                    if tokens[1] == 'classifyHeuristic':
                        parseClassifyHeuristic(path)
                    elif tokens[1] == 'classifyHeuristicRadius':
                        parseClassifyHeuristicRadius(path)
                    elif tokens[1] in ['distSPTD_certify', 'distributional', 'distSPTD_certify_attacked']:
                        parseClassifyDistributional(path)
                    elif tokens[1] == 'distSPTx_certify':
                        parseClassifyDistributionalJustPrediction(path)
                    elif tokens[1] == 'indivSPT_certify_attacked':
                        parseClassifyOnline(path)
                    elif tokens[1] in ['getErr', 'getErrSampling']:
                        parseGetErr(path)
                    elif tokens[1] == 'refinementExp':
                        parseRefinementExp(path)
            except:
                print("Invalid Format")
                traceback.print_exc()
        print('*' * 20)
