import os
a = (os.getcwd())
import sys

sys.path.insert(0, './code')
from functools import reduce
import algorithm
import algorithm_torch
import numpy as np
import pandas as pd
import pickle
import validate
import matplotlib.pyplot as plt
from tabulate import tabulate
from importlib import reload

def read_renal_data(surv_min=0):
  print('Read heart data')
  dat_gfr = pd.read_csv('joineRML_data/renal_gfr.csv')
  dat_haem = pd.read_csv('joineRML_data/renal_haem.csv')
  dat_surv = pd.read_csv('joineRML_data/renal_surv.csv')
  del dat_gfr[dat_gfr.columns[0]],dat_haem[dat_haem.columns[0]],dat_surv[dat_surv.columns[0]]
  #dat.head()
  idy = ['gfr', 'haematocrit']
  idx = ['gender', 'age','weight']
  id_col='id'
  surv_col='fuyears'
  status_col='failure'
  time_col='years'
  # dat = dat.dropna(subset = idy)
  # get id that has all biomarker measurements
  # id = dat[id_col].drop_duplicates().to_numpy()
  id = reduce(set.intersection, [set(dat_gfr[id_col].dropna()), set(dat_haem[id_col].dropna())])
  dat_gfr = dat_gfr[dat_gfr[id_col].isin(id)]
  dat_haem = dat_haem[dat_haem[id_col].isin(id)]
  dat_surv = dat_surv[dat_surv[id_col].isin(id)]
  n = len(id)
  # covariates sex and age
  dat_surv['gender'] = 1*(dat_surv['gender']=='male')
  X = np.array(dat_surv[np.concatenate(([id_col],idx))].drop_duplicates())[:, 1:]
  X = ((X.T - np.mean(X, axis = 0)[:,np.newaxis]) / np.std(X, axis = 0)[:,np.newaxis]).T
  # longitudinal biomarkers Y1,Y2,...
  nY = len(idy)
  Y = [[]] * nY
  time = [[]] * nY
  nn = [[]] * nY
  i=0
  dat0 = dat_gfr[[id_col, time_col, idy[i]]].dropna()
  Y[i] = np.array(dat0.groupby(id_col)[idy[i]].apply(lambda x: np.array(x)))
  Y[i] = np.array([(np.clip(x, 0, 1e5)) for x in Y[i]])
  Y[i] = (Y[i] - np.nanmean(np.concatenate(Y[i]))) / np.nanstd(np.concatenate(Y[i]))
  time[i] = np.array(dat0.groupby(id_col)[time_col].apply(lambda x: np.array(x)))
  nn[i] = np.array(list(map(len, time[i])))
  i=1
  dat0 = dat_haem[[id_col, time_col, idy[i]]].dropna()
  Y[i] = np.array(dat0.groupby(id_col)[idy[i]].apply(lambda x: np.array(x)))
  Y[i] = np.array([(np.clip(x, 0, 1e5)) for x in Y[i]])
  Y[i] = (Y[i] - np.nanmean(np.concatenate(Y[i]))) / np.nanstd(np.concatenate(Y[i]))
  time[i] = np.array(dat0.groupby(id_col)[time_col].apply(lambda x: np.array(x)))
  nn[i] = np.array(list(map(len, time[i])))
  Y = np.moveaxis(Y,0,1)
  time = np.moveaxis(time,0,1)
  nn = np.moveaxis(nn,0,1)
  # survival outcome: survival time and event indicator
  surv = np.array(dat_surv[[id_col, surv_col]].drop_duplicates())[:, 1:].flatten()
  status = np.array(dat_surv[[id_col, status_col]].drop_duplicates())[:, 1:].flatten()
  id_vec = np.where(np.array([np.prod([nn[i][j] >= 3 for j in range(nY)])*(surv[i]>=surv_min) for i in range(n)])==1)[0]
  n = len(id_vec)
  X = X[id_vec];Y = Y[id_vec];nn = nn[id_vec];time = time[id_vec];surv = surv[id_vec];status = status[id_vec]
  id = np.arange(n);
  surv_max = max(surv); print('surv_max='+str(surv_max))
  surv = surv / surv_max
  time = np.array([[time[i][j] / surv_max for j in range(nY)] for i in range(n)])
  dat = dict(id=id, time=time, surv=surv, nn=nn, status=status, X=X, Y=Y)
  ns = np.array([n, X.shape[1], Y.shape[1], 2, 1000, 100], dtype=np.int64)
  dat_train, ns_train, dat_test, ns_test = validate.split_data(dat,fold=4)
  print('Successfully read data: ' + str(ns_test[0]) + ' (25%) test patients; ' +
                                    str(ns_train[0]) + ' (75%) train patients.')
  return dat, ns, dat_train, dat_test, ns_train, ns_test

dat, ns, dat_train0, dat_test0, ns_train0, ns_test0 = read_renal_data(surv_min=0)
control = {'tol': 5e-3, 'condition': 'abs_diff', 'MCscaler': 2, 'MCmax': 2e5, 'EMmax': 50, 'trace_back': 1, 'test': False}
surv_max=19.219712526
num_knots=5
knots = np.quantile(np.concatenate([np.concatenate(x) for x in dat['time']]),q=np.arange(1,num_knots+1) / (num_knots+1))


name=''
lbda=[.1,.1];fold=4
#lbda = algorithm.select_lbda(dat_train0, ns_train0, knots=knots) #

ns0 = np.array([ss['n'], ss['nX'], ss['nY'], ss['nt'], ss['nMC'], 1000], dtype=np.int64)
dat = gen_data(ss, verbose=False, censor=True, q=1, seed=123)
dat_test = gen_data(ss, verbose=False, censor=False, q=1, seed=234)
knots = np.quantile(np.concatenate([np.concatenate(x) for x in dat['time']]),
                    q=np.arange(1, num_knots + 1) / (num_knots + 1))
lbda = algorithm.select_lbda(dat, ns0, knots=knots)  #

for num_knots in range(1,5):
    knots = np.arange(1, num_knots + 1) / (num_knots + 1)
    stage1est = algorithm.Stage1(dat_train0, ns_train0, lbda=lbda, knots=knots)
    stage2est_nonlinear = algorithm_torch.Stage2_torch(dat=dat_train0, ns0=ns_train0, stage1est=stage1est,
                                                       verbose=True, bandwidth=-.25, use_gpu='cuda:0',
                                                       knots=knots, filename='nonlinear_validate', control=control)
    stage2est_nonlinear = (validate.external_validation(dat_test0, stage2est_nonlinear, surv_max=surv_max))
    print(stage2est_nonlinear['validate']['auc']['final'])
    with open('nonlinear' + str(num_knots) + '_.pickle', 'wb') as handle:
        pickle.dump(stage2est_nonlinear, handle, protocol=pickle.HIGHEST_PROTOCOL)

