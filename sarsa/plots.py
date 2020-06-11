import matplotlib
import matplotlib.cm
import matplotlib.pyplot as pp
import numpy as np

from baselines.common import plot_util as pu
import os

from arguments import*
from tqdm import tqdm
import pickle

SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 22

pp.rc('font', size=SMALL_SIZE)          # controls default text sizes
pp.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
pp.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
pp.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pp.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pp.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
pp.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
pp.rcParams['legend.title_fontsize'] = SMALL_SIZE
pp.rcParams['axes.titlesize'] = BIGGER_SIZE

hps = list(enumerate((Jun6_10M())))

dont_cache = []

rewards = []
algos = []
lrs = []
beta = []
beta1 = []
eps = []
env_names = []


plot_envs = ["Qbert", "SpaceInvaders", 'Breakout', "BeamRider"]

pre = ""
means_cache = {}
if os.path.exists(pre + "mean_reward_cache.pkl"):
    means_cache = pickle.load(open(pre+ "mean_reward_cache.pkl", 'rb'))

for i, hp in tqdm(hps):
    log_dir = hp['log_dir']
    if not os.path.exists(log_dir):
        continue
    if log_dir in means_cache:
        r = means_cache[log_dir]
    else:
        try:
            result = pu.load_results(log_dir)
            if pre == "":
                r = result[0].monitor['r'].mean()
            else:
                r = result[0].monitor['r'].tail(1000).mean()
        except Exception as e:
            if log_dir in dont_cache:
                print(e, log_dir)
            continue
        if log_dir not in dont_cache:
            means_cache[log_dir] = r

    if not np.isfinite(r):
        #print("reward for",i, log_dir,"?", r)
        continue
    rewards.append(r)
    lrs.append(hp['lr'])
    beta.append(hp['beta_2'])
    beta1.append(hp['beta_1'])
    eps.append(hp.get('eps',1e-5))
    algos.append(hp['algo'])
    env_names.append(hp['env_name'][:-len('NoFrameskip-v4')])

pickle.dump(means_cache, open(pre + "mean_reward_cache.pkl",'wb'))


f, ax = pp.subplots(4,3,figsize=(15,20), constrained_layout=True)
for j, env_name in enumerate(plot_envs):
    for i, hpv, name, bounds in zip(range(3), [lrs, beta, eps], ['learning rate', 'beta', 'epsilon'],
                                    [(1e-5, 1), [0, 1], [1e-8, 10**(-1)]]):
        a = ax[j, i]
        if i == 0 and j == 0:
            a0 = a
        for alg in ['adam', 'tdprop', 'sgd']:
            if alg == 'sgd' and name != 'learning rate': continue
            idx = np.where((np.asarray(algos) == alg) * (np.asarray(env_names) == env_name))[0]
            a.scatter(np.asarray(hpv)[idx], np.asarray(rewards)[idx], label=alg)
        a.set_xlabel(name)
        a.set_xlim(*bounds)
        a.set_ylabel('avg reward')
        a.set_title(f'{env_name}')
        if name != 'beta':
            a.set_xscale('log')
handles, labels = a0.get_legend_handles_labels()
f.legend(handles, labels, loc='lower right')
pp.savefig(pre + f'hps_scatter_{"_".join(plot_envs)}.png')

f, ax = pp.subplots(1,4,figsize=(20, 5), constrained_layout=True)
for j, env_name in enumerate(plot_envs):
    hpv = lrs
    name = 'learning rate'
    bounds = (1e-5, 1)
    a = ax[j]
    for alg in ['adam', 'tdprop', 'sgd']:
        if alg == 'sgd' and name != 'learning rate': continue
        idx = np.where((np.asarray(algos) == alg) * (np.asarray(env_names) == env_name))[0]
        a.scatter(np.asarray(hpv)[idx], np.asarray(rewards)[idx], label=alg)
    a.set_xlabel(name)
    a.set_xlim(*bounds)
    a.set_ylabel('avg reward')
    a.set_title(f'{env_name}')
    if name != 'beta':
        a.set_xscale('log')
handles, labels = a0.get_legend_handles_labels()
f.legend(handles, labels, loc='lower right')
pp.savefig(pre + f'hps_scatter_lr_{"_".join(plot_envs)}.png')



nbins = 20
fidx = 0
ncl = 3
f, ax = pp.subplots(4,3,figsize=(15,20), constrained_layout=True)
for env_name in plot_envs:
    rmin = rmax = None
    for i, hpi, namei in zip(range(3), [lrs, beta, eps], ['learning rate', 'beta', 'epsilon']):
        for j, hpj, namej in zip(range(3), [lrs, beta, eps], ['learning rate', 'beta', 'epsilon']):
            if i == j or j < i:
                continue
            a = ax[fidx//ncl][fidx%ncl]
            idx1 = np.where((np.asarray(algos) == 'tdprop') * (np.asarray(env_names) == env_name))[0]
            idx2 = np.where((np.asarray(algos) == 'adam') * (np.asarray(env_names) == env_name))[0]
            x,y,z1 = np.asarray(hpi)[idx1], np.asarray(hpj)[idx1],np.asarray(rewards)[idx1]
            x2,y2,z2 = np.asarray(hpi)[idx2], np.asarray(hpj)[idx2],np.asarray(rewards)[idx2]
            if not 'beta' in namei:
                xbins = np.logspace(np.log(x.min()),np.log(x.max()),nbins,base=np.e)
                a.set_xscale('log')
            else:
                xbins = np.linspace(x.min(),x.max(),nbins)
            if not 'beta' in namej:
                ybins = np.logspace(np.log(y.min()),np.log(y.max()),nbins,base=np.e)
                a.set_yscale('log')
            else:
                ybins = np.linspace(y.min(),y.max(),nbins)
            counts, xbins, ybins = np.histogram2d(x, y, bins=(xbins, ybins))
            sums, xbins, ybins = np.histogram2d(x, y, weights=z1, bins=(xbins, ybins))
            q = (sums / counts)

            xs, ys, zs = [],[],[]
            for k in range(len(xbins)-1):
                for l in range(len(ybins)-1):
                    if counts[k,l] > 0:
                        xs.append(xbins[k])
                        ys.append(ybins[l])
                        zs.append(q[k, l])
            if np.var(ys) == 0 or np.var(xs) == 0:
                continue
            zs1 = zs
            sums, xbins, ybins = np.histogram2d(x2, y2, weights=z2, bins=(xbins, ybins))
            q = (sums / counts)
            zs2 = []
            for k in range(len(xbins)-1):
                for l in range(len(ybins)-1):
                    if counts[k,l] > 0:
                        zs2.append(q[k, l])
            zs = np.float32(zs1) - np.float32(zs2)
            if rmin is None:
                rmin_ = zs.min()
                rmax_ = zs.max()
                rmin = -max(abs(rmin_), abs(rmax_))
                rmax = -rmin

            norm = matplotlib.colors.Normalize(rmin, rmax)
            im = a.tricontourf(xs, ys, zs, norm=norm, cmap=matplotlib.cm.bwr, 
                    levels=np.linspace(rmin,rmax, 11))
            #im = a.tricontourf(xs, ys, zs, norm=norm, cmap=matplotlib.cm.bwr)
            #a.pcolormesh(*np.meshgrid(xbins, ybins), (sums / counts).T)
            fidx += 1
            if fidx % ncl == 0:
                pp.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=im.cmap), ax=a)
            a.set_xlabel(namei)
            a.set_ylabel(namej)
            a.set_title(f'{env_name}')

pp.savefig(pre + f'hp_pairs_histo_diff_fill_{"_".join(plot_envs)}.png')

if 0:
    nbins = 20
    fidx = 0
    ncl = 3
    f, ax = pp.subplots(4,3,figsize=(10,10))
    for env_name in plot_envs:
        renv = np.asarray(rewards)[np.where(np.asarray(env_names) == env_name)[0]]
        rmin, rmax = np.min(renv), np.max(renv)
        for alg in ['adam', 'tdprop']:
            for i, hpi, namei in zip(range(3), [lrs, beta, eps], ['learning rate', 'beta2', 'eps']):
                for j, hpj, namej in zip(range(3), [lrs, beta, eps], ['learning rate', 'beta2', 'eps']):
                    if i == j or j < i:
                        continue
                    a = ax[fidx//ncl][fidx%ncl]
                    idx = np.where((np.asarray(algos) == alg) * (np.asarray(env_names) == env_name))[0]
                    x,y,z = np.asarray(hpi)[idx], np.asarray(hpj)[idx],np.asarray(rewards)[idx]
                    if namei != 'beta2':
                        xbins = np.logspace(np.log(x.min()),np.log(x.max()),nbins,base=np.e)
                        a.set_xscale('log')
                    else:
                        xbins = np.linspace(x.min(),x.max(),nbins)
                    if namej != 'beta2':
                        ybins = np.logspace(np.log(y.min()),np.log(y.max()),nbins,base=np.e)
                        a.set_yscale('log')
                    else:
                        ybins = np.linspace(y.min(),y.max(),nbins)
                    counts, xbins, ybins = np.histogram2d(x, y, bins=(xbins, ybins))
                    sums, xbins, ybins = np.histogram2d(x, y, weights=z, bins=(xbins, ybins))
                    q = (sums / counts)

                    xs, ys, zs = [],[],[]
                    for k in range(len(xbins)-1):
                        for l in range(len(ybins)-1):
                            if counts[k,l] > 0:
                                xs.append(xbins[k])
                                ys.append(ybins[l])
                                zs.append(q[k, l])
                    a.tricontourf(xs, ys, zs, norm=matplotlib.colors.Normalize(rmin, rmax))
                    #a.pcolormesh(*np.meshgrid(xbins, ybins), (sums / counts).T)
                    fidx += 1
                    a.set_xlabel(namei)
                    a.set_ylabel(namej)
                    a.set_title(f'{alg} {env_name}')

    pp.tight_layout()
    pp.savefig(f'hp_pairs_histo_fill_{"_".join(plot_envs)}.png')
