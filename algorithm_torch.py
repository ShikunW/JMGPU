# -*- coding: utf-8 -*-
;==========================================
; Title:  algorithm for a joint model with GPU support
; Author: Shikun Wang
; Date:   1 Dec 2020
;==========================================
"""
    algorithm_torch.py defines all functions to accelerate by CPU / GPU
"""

# import useful only for development
import numpy as np
import pickle
import timeit, math
from scipy.linalg import block_diag
import torch

from algorithm import Convergence, calc_initial_value, \
    Estep_gen_c_pos, EM_prepare, \
    Mstep_beta, Mstep_gamma, \
    alpha_iteration_number, \
    likelihood, \
    evaluate

# batch_size = 100
# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda:0' if use_cuda else 'cpu')
def Estep_alpha_torch(alpha_tensor, gamma_tensor, X_tensor, Bt_tensor, Vt_tensor, Vt2_tensor, fTdc_tensor,
                      mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor, ns):
    """Calculate Estep for alpha (Sec 2.2.7)

    Args:
        alpha: a vector of old alpha.
        gamma: a vector of old gamma.
        X: a matrix of time-independent covariates, dim = n*nX.
        Bt: 1/2(min{Ti,dk+1}−dk−1)I{Ti>=dk−1}I(min{Ti,dk} in Ij)B(min{Ti,dk}), dim = n*K*J.
        Vt: oplus [1,t], dim = n*K*(2nY).
        mu_c_pos, posterior mean of random effects c, dim = n*(2nY).
        Sc_pos_sqrt, posterior variance of random effects c, dim = n*(2nY)*(2nY).
        Z, ~MVN(0,1) for sampling random effects c, dim = nMC*nMC.
        ns: a vector of the size parameters.

    Returns:
        a matrix of row bind of I(alpha) and S(alpha)
    """
    # beta = beta_new;alpha = alpha_new; gamma = gamma_new
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns
    nt2 = min(nt, 2)
    idx = np.arange((nt2 + q) * nY).reshape((nY, nt2 + q))
    idx1 = idx[:,np.arange(nt2)].flatten()
    idx2 = idx[:,np.arange(nt2, nt2+q)].flatten()
    alpha0 = alpha_tensor[np.arange(nX)]
    alpha1 = alpha_tensor[np.arange(nX, alpha_tensor.shape[0])]
    # out = np.zeros((nX + nY + 1, nX + nY))
    Bg = torch.matmul(Bt_tensor, gamma_tensor).unsqueeze(1)
    c = torch.matmul(Sc_pos_sqrt_tensor.unsqueeze(1), Z_tensor.unsqueeze(0).unsqueeze(-1)).squeeze(3) + \
        mu_c_pos_tensor.unsqueeze(1)
    b = c[:, :, idx1] # ???
    Gb = torch.matmul(Vt_tensor.unsqueeze(1), b.unsqueeze(2).unsqueeze(-1)).squeeze(4)
    if q > 0:
        u = c[:,:,idx2] # ???
        Wu = torch.matmul(Vt2_tensor.unsqueeze(1), \
                          u.unsqueeze(2).unsqueeze(-1)).squeeze(4)
        Gb = Gb + Wu
    Gba = torch.matmul(Gb, alpha1)
    Xa = torch.matmul(X_tensor, alpha0).reshape(n, 1, 1)
    itg = Bg * torch.exp(Xa + Gba) * fTdc_tensor.reshape(n, nMC, 1)

    itg1 = itg.reshape(n, nMC, K, 1)
    itg2 = itg.reshape(n, nMC, K, 1, 1); del itg
    X_rep = X_tensor.unsqueeze(1).expand(n, nMC, nX)
    X_rep = X_rep.unsqueeze(2).expand(n, nMC, K, nX)
    concat1 = torch.cat((X_rep, Gb), axis=-1)
    concat2 = torch.matmul(concat1.unsqueeze(-1), concat1.unsqueeze(-2)) # easy to be out of memory
    out1 = itg1 * concat1; del itg1, concat1
    out2 = itg2 * concat2; del itg2, concat2
    out1 = torch.sum(out1, (0, 1, 2))
    out2 = torch.sum(out2, (0, 1, 2))
    #del itg1, itg2, concat1, concat2
    torch.cuda.empty_cache()
    return out1, out2

def Estep_gamma_batch_torch(alpha_tensor, X_tensor, Vt_tensor, Vt2_tensor, dit_tensor, fTdc_tensor,
                            mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor, device, ns):
    """Calculate batch Estep for gamma (Sec 2.2.6)
    Args:
        alpha: a vector of old alpha.
        X: a matrix of time-independent covariates, dim = n*nX.
        Vt: oplus [1,t], dim = n*K*(2nY).
        Vt2: oplus [1,polynomial(t)], dim = n*K*(q*nY).
        dit: 1/2(min{Ti,dk+1}−dk−1)I{Ti>=dk−1}I(min{Ti,dk} in Ij), first row in equation (13), dim = n*K*J.
        fTdc: f(c|Y,T,delta;theta) at current step, dim = n*nMC*(2nY).
        mu_c_pos, posterior mean of random effects c, dim = n*(2nY).
        Sc_pos_sqrt, posterior variance of random effects c, dim = n*(2nY)*(2nY).
        Z, ~Normal(0,1) for sampling random effects c, dim = nMC*nMC.
        ns: a vector of the size parameters.

    Returns:
        a vector of Egamma, the denominator of equation (13)
    """
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns

    Egamma = torch.zeros(J).to(device)

    ns_epoch = ns.copy()
    ns_epoch[0] = batch_size
    batch_num = math.ceil(n / batch_size)
    for batch_i in range(batch_num): # multiple GPU?
        if batch_i == batch_num - 1:
            idx_epoch = range(ns_epoch[0] * batch_i, n)
            ns_epoch[0] = len(idx_epoch)
        else:
            idx_epoch = range(ns_epoch[0]) + ns_epoch[0] * batch_i
        Egamma += Estep_gamma_torch(alpha_tensor, X_tensor[idx_epoch], Vt_tensor[idx_epoch], Vt2_tensor[idx_epoch],
                                    dit_tensor[idx_epoch], fTdc_tensor[idx_epoch],
                                    mu_c_pos_tensor[idx_epoch], Sc_pos_sqrt_tensor[idx_epoch], Z_tensor, ns_epoch)
        torch.cuda.empty_cache()
    return Egamma


def Estep_gamma_torch(alpha_tensor, X_tensor, Vt_tensor, Vt2_tensor, dit_tensor, fTdc_tensor,
                      mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor, ns):
    """Calculate Estep for gamma (Sec 2.2.6)
    Args:
        alpha: a vector of old alpha.
        X: a matrix of time-independent covariates, dim = n*nX.
        Vt: oplus [1,t], dim = n*K*(2nY).
        dit: 1/2(min{Ti,dk+1}−dk−1)I{Ti>=dk−1}I(min{Ti,dk} in Ij), first row in equation (13), dim = n*K*J.
        fTdc: f(c|Y,T,delta;theta) at current step, dim = n*nMC*(2nY).
        mu_c_pos, posterior mean of random effects c, dim = n*(2nY).
        Sc_pos_sqrt, posterior variance of random effects c, dim = n*(2nY)*(2nY).
        Z, ~MVN(0,1) for sampling random effects c, dim = nMC*nMC.
        ns: a vector of the size parameters.

    Returns:
        a vector of Egamma, the denominator of equation (13)
    """
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns
    nt2 = min(nt, 2)
    idx = np.arange((nt2 + q) * nY).reshape((nY, nt2 + q))
    idx1 = idx[:, np.arange(nt2)].flatten()
    idx2 = idx[:, np.arange(nt2, nt2+q)].flatten()
    alpha0 = alpha_tensor[np.arange(nX)]
    alpha1 = alpha_tensor[range(nX, nX + nY)]
    c = torch.matmul(Sc_pos_sqrt_tensor.unsqueeze(1), Z_tensor.unsqueeze(0).unsqueeze(-1)).squeeze(3) + \
        mu_c_pos_tensor.unsqueeze(1) # equavelent to: sample c_is ~ MVN(mu_pos, Sigma_pos)
    b = c[:, :, idx1] # ???

    Gb = torch.matmul(Vt_tensor.unsqueeze(1), b.unsqueeze(2).unsqueeze(-1)).squeeze(4)
    if q > 0:
        u = c[:, :, idx2] # ???
        Wu = torch.matmul(Vt2_tensor.unsqueeze(1), \
                          u.unsqueeze(2).unsqueeze(-1)).squeeze(4)
        Gba = torch.matmul(Gb + Wu, alpha1)
    else:
        Gba = torch.matmul(Gb, alpha1)
    Xa = torch.matmul(X_tensor, alpha0).reshape(n, 1, 1)
    expf = torch.exp(Xa + Gba) * fTdc_tensor.unsqueeze(-1)
    out3 = torch.sum(torch.sum(torch.sum(expf.unsqueeze(-1) * dit_tensor.unsqueeze(1), axis=0), axis=0), axis=0)
    torch.cuda.empty_cache()
    return out3


def Estep_fTdc_torch(alpha_tensor, gamma_tensor, X_tensor, dVT_tensor, dVT2_tensor, Bt_tensor, Vt_tensor, Vt2_tensor,
                     mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor, ns):
    """Calculate Estep for gamma (Sec 2.2.6)
        Args:
        alpha: a vector of old alpha.
        X: a matrix of time-independent covariates, dim = n*nX.
        Vt: oplus [1,t], dim = n*K*(2nY).
        dit: 1/2(min{Ti,dk+1}−dk−1)I{Ti>=dk−1}I(min{Ti,dk} in Ij), first row in equation (13), dim = n*K*J.
        fTdc: f(c|Y,T,delta;theta) at current step, dim = n*nMC*(2nY).
        mu_c_pos, posterior mean of random effects c, dim = n*(2nY).
        Sc_pos_sqrt, posterior variance of random effects c, dim = n*(2nY)*(2nY).
        Z, ~MVN(0,1) for sampling random effects c, dim = nMC*nMC.
        ns: a vector of the size parameters.

        Returns:
        a vector of Egamma, the denominator of equation (13)
    """
    # alpha_tensor = alpha_old_tensor; gamma_tensor = gamma_old_tensor
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns

    #idx1 = np.arange(nt * nY)
    #idx2 = np.arange(nt * nY, (nt + q) * nY)

    nt2 = min(nt, 2)
    idx = np.arange((nt2 + q) * nY).reshape((nY, nt2 + q))
    idx1 = idx[:,np.arange(nt2)].flatten()
    idx2 = idx[:,np.arange(nt2, nt2+q)].flatten()

    alpha0 = alpha_tensor[range(nX)]
    alpha1 = alpha_tensor[range(nX, nX + nY)]
    c = torch.matmul(Sc_pos_sqrt_tensor.unsqueeze(1),
                     Z_tensor.unsqueeze(0).unsqueeze(-1)).squeeze(3) + \
        mu_c_pos_tensor.unsqueeze(1)

    b = c[:, :, idx1] # ???
    Gb = torch.matmul(Vt_tensor.unsqueeze(1), b.unsqueeze(2).unsqueeze(-1)).squeeze(4)
    if q > 0:
        u = c[:, :, idx2] # ???
        Wu = torch.matmul(Vt2_tensor.unsqueeze(1), u.unsqueeze(2).unsqueeze(-1)).squeeze(4)
        Gba = torch.matmul(Gb + Wu, alpha1)
    else:
        Gba = torch.matmul(Gb, alpha1)
    Xa = torch.matmul(X_tensor, alpha0).reshape(n, 1, 1)
    Bg = torch.matmul(Bt_tensor, gamma_tensor).unsqueeze(1)
    itg = torch.sum(Bg * torch.exp(Xa + Gba), axis = 2)
    wz = torch.matmul(dVT_tensor.unsqueeze(1), b.unsqueeze(-1)).squeeze(3)
    if q > 0:
        wz2 = torch.matmul(dVT2_tensor.unsqueeze(1), u.unsqueeze(-1)).squeeze(3)
        wza = torch.matmul(wz + wz2, alpha1)
    else:
        wza = torch.matmul(wz, alpha1)
    fTdc = torch.exp(wza - itg)
    del Bg, Xa, wz, wza, itg
    torch.cuda.empty_cache()
    return fTdc


def EM_alpha_torch(alpha_tensor, gamma_tensor, X_tensor, dXsum, dVT, dVT2, Bt_tensor, Vt_tensor, Vt2_tensor,
                   fTdc_tensor, mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor, ns, control, device):
    """Calculate Estep for gamma (Sec 2.2.6)
    Args:
        alpha: a vector of old alpha.
        gamma: a vector of old gamma.
        X: a matrix of time-independent covariates, dim = n*nX.
        status: a vector of censor indicator, dim = n
        dXsum: sum(status * X), dim = nX
        ZT: oplus [1,T], dim = n*nY*(2nY)???
        Bt: 1/2(min{Ti,dk+1}−dk−1)I{Ti>=dk−1}I(min{Ti,dk} in Ij)B(min{Ti,dk}), dim = n*K*J.
        Vt: oplus [1,t], dim = n*K*(2nY).
        fTdc: f(c|Y,T,delta;theta) at current step, dim = n*nMC*(2nY).
        mu_c_pos: posterior mean of random effects c, dim = n*(2nY).
        Sc_pos_sqrt: posterior variance of random effects c, dim = n*(2nY)*(2nY).
        Z, ~MVN(0,1) for sampling random effects c, dim = nMC*nMC.
        ns: a vector of the size parameters.
        niter: number of maximum allowed newton-raphson iteration

    Returns:
        a vector of Egamma, the denominator of equation (13)
    """
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns
    if q > 0:
        dGc = np.sum([np.dot(np.column_stack((dVT[i], dVT2[i])),
                             ((torch.matmul(Sc_pos_sqrt_tensor[i], Z_tensor.T) + mu_c_pos_tensor[i][:, np.newaxis]) *
                              fTdc_tensor[i][np.newaxis, :]).sum(axis=1).cpu().numpy())
                      for i in range(n)], axis=0)
    else:
        dGc = np.sum([np.dot(dVT[i],
                             ((torch.matmul(Sc_pos_sqrt_tensor[i], Z_tensor.T) + mu_c_pos_tensor[i][:, np.newaxis]) *
                              fTdc_tensor[i][np.newaxis, :]).sum(axis=1).cpu().numpy())
                      for i in range(n)], axis=0)
    first = np.concatenate((dXsum, dGc))
    first_tensor = torch.FloatTensor(first).to(device)

    alpha_new_tensor = alpha_tensor

    history_alpha = {'logLik': [0, 0],
               'param': [alpha_tensor.cpu().numpy()] * control['trace_back'],
               'logLik_rel_ind': [0],
               'param_ind': [np.zeros(len(alpha_tensor))] * control['trace_back'],
               'param_abs_ind': [np.zeros(len(alpha_tensor))] * control['trace_back'],
               'param_rel_ind': [np.zeros(len(alpha_tensor))] * control['trace_back']}
    while True:
        # print("alpha =", alpha_new.round(3))
        alpha_old_tensor = alpha_new_tensor

        # split the data into 10 parts, and calculate out1, out2 sequentially
        # need to apply distributed computing technique to parallelize

        out1 = torch.zeros(nX + nY).to(device)
        out2 = torch.zeros((nX + nY, nX + nY)).to(device)

        ns_epoch = ns.copy(); ns_epoch[0] = batch_size
        batch_num = math.ceil(n / batch_size)
        for batch_i in range(batch_num):
            if batch_i == batch_num - 1:
                idx_epoch = range(ns_epoch[0] * batch_i, n)
                ns_epoch[0] = len(idx_epoch)
            else:
                idx_epoch = range(ns_epoch[0]) + ns_epoch[0] * batch_i
            torch.cuda.empty_cache()
            out1_epoch, out2_epoch = Estep_alpha_torch(alpha_old_tensor, gamma_tensor, X_tensor[idx_epoch],
                                     Bt_tensor[idx_epoch], Vt_tensor[idx_epoch], Vt2_tensor[idx_epoch],
                                     fTdc_tensor[idx_epoch],
                                     mu_c_pos_tensor[idx_epoch], Sc_pos_sqrt_tensor[idx_epoch], Z_tensor, ns_epoch)

            out1 += out1_epoch
            out2 += out2_epoch
            torch.cuda.empty_cache()

        Sa = (first_tensor - out1)[:, None]
        Ia = out2
        alpha_new_tensor = alpha_old_tensor + torch.matmul(torch.inverse(Ia), Sa).flatten()
        history_alpha['param'].append(alpha_new_tensor.cpu().numpy())
        if Convergence(history=history_alpha, verbose=False, control=control)[0]:
            break
    torch.cuda.empty_cache()
    return alpha_new_tensor


def Estep_prepare_torch(beta, alpha_tensor, gamma_tensor, SinvDSY, SinvDSU, X_tensor, dVT_tensor, dVT2_tensor,
                        Bt_tensor, Vt_tensor, Vt2_tensor, Sc_pos_sqrt_tensor, ns, device):
    """Update Estep values for EM (Sec 2.2.3)

    Args:
        beta: pre-calculated sum(U'U)^{-1}, dim = (nX+2)*(nX+2)
        UY: pre-calculated sum(U'Y), dim = (nX+2)*1
        UV: pre-calculated sum(U'V)^{-1}, dim = (nX+2)*2
        fTdc: f(c|Y,T,delta;theta) at current step, dim = n*nMC*(nY*2)
        mu_c_pos, posterior mean of random effects c, dim = n*(nY*2)
        Sc_pos_sqrt, posterior variance of random effects c, dim = n*(nY*2)*(nY*2)
        Z, ~MVN(0,1) for sampling random effects c, dim = nMC*nMC`

    Returns:
    posterior sampling mean and a standard normal matrix for getting posterior c; f(T_i, delta_i | c_is; theta)

    """
    # beta, alpha_tensor, gamma_tensor = beta_old, alpha_old_tensor, gamma_old_tensor
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns
    np.random.seed(0)
    mu_c_pos, Z = Estep_gen_c_pos(beta, SinvDSY, SinvDSU, ns)

    mu_c_pos_tensor = torch.FloatTensor(mu_c_pos).to(device)
    Z_tensor = torch.FloatTensor(Z).to(device)
    fTdc_tensor0 = torch.zeros((n, nMC)).to(device)

    # split the data into 10 parts, and calculate fTdc sequentially
    # need to apply distributed computing technique to parallelize

    ns_epoch = ns.copy(); ns_epoch[0] = batch_size
    batch_num = math.ceil(n / batch_size)
    for batch_i in range(batch_num):
        if batch_i == batch_num - 1:
            idx_epoch = range(ns_epoch[0] * batch_i, n)
            ns_epoch[0] = len(idx_epoch)
        else:
            idx_epoch = range(ns_epoch[0]) + ns_epoch[0] * batch_i

        fTdc_tensor0[idx_epoch] = Estep_fTdc_torch(alpha_tensor, gamma_tensor, X_tensor[idx_epoch],
                                                  dVT_tensor[idx_epoch], dVT2_tensor[idx_epoch],
                                                  Bt_tensor[idx_epoch], Vt_tensor[idx_epoch],
                                                  Vt2_tensor[idx_epoch],
                                                  mu_c_pos_tensor[idx_epoch],
                                                  Sc_pos_sqrt_tensor[idx_epoch], Z_tensor, ns_epoch)
        torch.cuda.empty_cache()

    # the above fTdc_tensor must be equal to this!
    # fTdc_tensor = Estep_fTdc_torch(alpha_tensor, gamma_tensor, X_tensor, dVT_tensor, Bt_tensor, Vt_tensor, mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor, ns)

    fTdc_tensor = torch.stack(list(map(lambda x: (x + 1e-10) / torch.sum(x + 1e-10), fTdc_tensor0)))
    #fTdc_denom = torch.sum(torch.log(torch.stack(list(map(lambda x: torch.mean(x), fTdc_tensor))))).cpu().numpy()
    #print('fTdc_tensor=',fTdc_tensor)
    fTdc_denom = torch.sum(torch.log(torch.mean(fTdc_tensor0, axis=1))).cpu().numpy()
    return mu_c_pos, Z, mu_c_pos_tensor, Z_tensor, fTdc_tensor, fTdc_denom



def Stage2_torch(dat, ns0, stage1est, verbose=False, bandwidth=.25, knots=np.empty(0),
                 use_gpu=True, filename = 'unnamed_JM_result',
                 control= {'tol': 5e-3, 'condition': 'rel_diff', 'MCscaler': 2,
                           'MCmax': 2e5, 'EMmax': 200, 'trace_back': 3, 'test': False}):
    """Stage 2 estiamtes for mean components (Sec 2.2.2)

    Args:
        dat: a dictionary of all data.
        ns0: a vector of the size parameters.
        stage1est: a dict of parameters from stage 1 estimate.
        verbose: = True, print details in each EM iteration.
        bandwidth: >0, bandwidth for piecewise constant baseline hazard.
        knots: a vector of knots for nonlinear model.
        batch_size: number of subject in each batch.
        control: convergence parameters and criteria

    Returns:
       a dict of point estimates for parameters, computation time, history

    Raises:

    """
    elapse = {}
    start_time = timeit.default_timer()  # start to time the whole stage2 algorithm

    # calculate initial values for beta, alpha, gamma, baseline hazard cut points a
    beta_init, alpha_init, gamma_init, a = calc_initial_value(dat, ns0, stage1est['beta'], bandwidth)

    beta_new, alpha_new, gamma_new = beta_init, alpha_init, gamma_init
    para_new = np.concatenate([beta_new, alpha_new, gamma_new])
    # get data
    X = dat['X']; #Y = dat['Y']; status = dat['status']; nn = dat['nn']# ; time = dat['time']

    elapse['calc_initial_value'] = timeit.default_timer() - start_time  # end to time calc_initial_value

    start_time0 = timeit.default_timer()  # start time EM_prepare

    # calculate useful values for EM
    UUinv, UY, UV, dX, dXsum, dVT, VT, BT, dBTsum, Sc_pos_sqrt, SinvDSY, SinvDSU, \
    d, dit, Bt, Vt, Va, Vt2, dVT2, Y, U, VlogLik, Vpos_inv= EM_prepare(dat, stage1est, a, ns0, knots)
    
    J = len(a) - 1  # number of gamma's for baseline hazard
    K = len(d) - 2  # number of pieces for trapozoidal rule
    q = len(knots) # number of knots for truncated polynomial
    #n, nX, nY, nt, nMC = ns0
    ns = np.insert(ns0, len(ns0), [q, J, K]) # update the sample size vector
    elapse['EM_prepare'] = timeit.default_timer() - start_time0  # end to time EM_prepare
    # print('EM_prepare costs', elapsed0, 's')

    # decide to use GPU or CPU
    use_cuda = ((use_gpu!=False) and torch.cuda.is_available())
    device = torch.device(use_gpu if use_cuda else 'cpu') # 'cuda:0'
    if verbose:
        print("use_cuda=", use_cuda, "; device=", device)

    # send useful values to device
    X_tensor = torch.FloatTensor(X).to(device)
    dVT_tensor = torch.FloatTensor(dVT).to(device)
    dBTsum_tensor = torch.FloatTensor(dBTsum).to(device)
    Bt_tensor = torch.FloatTensor(Bt).to(device)
    Vt_tensor = torch.FloatTensor(Vt).to(device)
    dit_tensor = torch.FloatTensor(dit).to(device)
    Sc_pos_sqrt_tensor = torch.FloatTensor(Sc_pos_sqrt).to(device)
    alpha_new_tensor = torch.FloatTensor(alpha_new).to(device)
    gamma_new_tensor = torch.FloatTensor(gamma_new).to(device)

    dVT2_tensor = torch.FloatTensor(dVT2).to(device)
    Vt2_tensor = torch.FloatTensor(Vt2).to(device)
    UV_tensor = torch.FloatTensor(UV).to(device)


    # initialize temporary parameters to decide whether to increase nMC or not
    delta_rel = np.zeros(3)  # initial values for coefficient of variation score
    coef_var_old = 0  # initial coefficient of variation score
    # track history of parameters, log-likelihoods, and elapsed time

    history = {'logLik': [-9999999999999],
               'param': [para_new],
               'nMC': [ns[4]],
               'logLik_rel_ind': [0],
               'param_ind': [np.zeros(len(para_new))],
               'param_abs_ind': [np.zeros(len(para_new))],
               'param_rel_ind': [np.zeros(len(para_new))]}
    control_alpha = control.copy()
    elapse['EM'] = []  # initialize time array for EM

    while True:
        elapse_this = [ns[4]]  # initialize iter-th iteration elapsed time, first element is current nMC
        nMC = int(ns[4].copy())
        alpha_old_tensor = alpha_new_tensor
        gamma_old_tensor = gamma_new_tensor
        #alpha_old = alpha_new
        #gamma_old = gamma_new
        beta_old = beta_new
        # logLik_old=logLik_new;logLik_count=0

        # calculate posterior mean of random effects, a random matrix from standard normal,
        # and f(T,delta|c,theta) for Estep (Sec 2.2.3) equation (11)
        start_time0 = timeit.default_timer()  # start to time Estep_prepare
        mu_c_pos, Z, mu_c_pos_tensor, Z_tensor, fTdc_tensor, fTdc_denom = \
            Estep_prepare_torch(beta_old, alpha_old_tensor, gamma_old_tensor, SinvDSY, SinvDSU,
                                X_tensor, dVT_tensor, dVT2_tensor, Bt_tensor, Vt_tensor, Vt2_tensor,
                                Sc_pos_sqrt_tensor, ns, device)
        # fTdc = fTdc_tensor.cpu().numpy()
        elapsed0 = timeit.default_timer() - start_time0; elapse_this.append(elapsed0)  # end to time Estep_prepare
        # print('Estep_prepare_numba costs', elapsed0, 's')

        logLik_new = likelihood(VlogLik, Vpos_inv, beta_new, Y, U, fTdc_denom)
        history['logLik'].append(logLik_new)

        # Update gamma
        # calculate Estep for gamma
        start_time0 = timeit.default_timer()  # start to time EMstep_gamma

        # split the data into 10 parts, and calculate Egamma sequentially
        # need to apply distributed computing technique to parallelize

        Egamma_tensor = Estep_gamma_batch_torch(alpha_old_tensor, X_tensor, Vt_tensor,
                                          Vt2_tensor, dit_tensor, fTdc_tensor,
                                          mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor, device, ns)
        gamma_new_tensor = Mstep_gamma(dBTsum_tensor, Egamma_tensor)
        gamma_new = gamma_new_tensor.cpu().numpy()
        elapsed0 = timeit.default_timer() - start_time0; elapse_this.append(elapsed0)  # end to time Mstep_gamma
        # print('EMstep_gamma_numba costs', elapsed0, 's')

        # Update alpha
        # calculate EM for alpha
        # decide max number of newton-raphson iteration
        conv_proportion = np.mean(history['param_ind'][-1])
        niter_alpha = alpha_iteration_number(conv_proportion=conv_proportion, type='dynamic')

        start_time0 = timeit.default_timer()  # start to time EM_alpha
        control_alpha['EMmax'] = niter_alpha# = {'tol': 5e-3, 'condition': 'rel_diff', 'EMmax': , 'trace_back': 1}
        alpha_new_tensor = EM_alpha_torch(alpha_old_tensor, gamma_new_tensor, X_tensor, dXsum, dVT, dVT2,
                                          Bt_tensor, Vt_tensor, Vt2_tensor, fTdc_tensor,
                                          mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor,
                                          ns, control_alpha, device)
        alpha_new = alpha_new_tensor.cpu().numpy()
        elapsed0 = timeit.default_timer() - start_time0; elapse_this.append(elapsed0)  # end to time Mstep_alpha
        # print('EM_alpha_numba costs', elapsed0, 's')

        # Update beta
        # calculate Mstep for beta
        start_time0 = timeit.default_timer()  # start to time Mstep_beta
        beta_new = Mstep_beta(UUinv, UY, UV_tensor, fTdc_tensor, mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor)
        #beta_new = stage1est['beta'].T.flatten()
        elapsed0 = timeit.default_timer() - start_time0; elapse_this.append(elapsed0)  # end to time Mstep_beta
        # print('Mstep_beta costs', elapsed0, 's')

        elapse['EM'].append(elapse_this)  # record elapsed time details
        # decide if converge?
        para_new = np.concatenate([beta_new, alpha_new, gamma_new])
        history['param'].append(para_new)
        Conv, delta, convind, history = Convergence(history=history, control=control, verbose=verbose)
        torch.cuda.empty_cache()
        if Conv:
            break
        delta_rel = np.append(np.delete(delta_rel, 0), delta)
        # increase the number of MC point every 5 ieration
        coef_var_new = control['test'] * (1 * ((len(history['logLik'])-1) % 5 == 0)) + \
                       (1-control['test']) * np.std(delta_rel) / np.mean(delta_rel)

        #print(coef_var_new, coef_var_old)
        if coef_var_new > coef_var_old:
            if ns[4] < 2e4: # control['MCmax']
                ns4_old = ns[4].copy()
                ns[4] += int(ns[4] / control['MCscaler'])

            if ((ns4_old < 1e3) and (ns[4] > 1e3)) or ((ns4_old < 1e4) and (ns[4] > 1e4)): #
                ns[5] /= 10
        print("ns[5]=", ns[5])
        history['nMC'].append(ns[4])
        #print(history['nMC'])
            #print('Add MC counts, current count is',ns[4])
        coef_var_old = coef_var_new

        # print current updates if verbose = True
        if verbose:
            print("beta =", beta_new.flatten().round(3))
            print("alpha =", alpha_new.round(3))
            print("gamma =", gamma_new.round(3))
            print("---------- Iter #" + str(len(history['param'])) + " finished, elapsed time=" + str(np.round(timeit.default_timer() - start_time,1)) + "s ---------")
        stage2est = {'beta': beta_new, 'alpha': alpha_new, 'gamma': gamma_new,
                     'sigma_e': stage1est['sigma_e'], 'sigma_u': stage1est['sigma_u'],
                     'Sigma_b': stage1est['Sigma_b'], 'knots': knots,
                     'MC_pts': [ns0[4], ns[4]], 'history': history,
                     'elapse': elapse, 'a': a, 'ns': ns} # , 'Yhat': Yhat

        if len(filename) > 0:
            with open(filename+'_details.pickle', 'wb') as handle:
                pickle.dump(stage2est, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = timeit.default_timer() - start_time # end time the whole stage 2 estimate

    # print the final estimates of mean components if verbose = True
    if verbose:
        print("Converges at", len(history['logLik']), "th iteration\nElapsed time =", round(elapsed, 2), "s.")
        print("beta =", beta_new.flatten().round(3))
        print("alpha =", alpha_new.round(3))
        print("gamma =", gamma_new.round(3))
        print('MC counts start =', nMC, '; end =', ns[4])
    

    start_time = timeit.default_timer() # start to time longitudinal prediction

    elapsed = timeit.default_timer() - start_time  # end time longitudinal prediction

    elapse['prediction'] = elapsed
    # prepare the output
    stage2est = {'beta_init': beta_init, 'alpha_init': alpha_init, 'gamma_init': gamma_init,
                 'beta': beta_new, 'alpha': alpha_new, 'gamma': gamma_new,
                 'sigma_e': stage1est['sigma_e'], 'sigma_u': stage1est['sigma_u'],
                 'Sigma_b': stage1est['Sigma_b'], 'knots': knots,
                 'Iteration': iter, 'MC_pts': [ns0[4], ns[4]], 'likelihood': logLik_new,
                 'elapse': elapse, 'a': a, 'history': history,
                 'ns': ns}

    # evaluate model
    stage2est['performance'] = evaluate(ns, stage2est, logLik_new)
    
    if len(filename)>0:
        with open(filename + '_final.pickle', 'wb') as handle:
            pickle.dump(stage2est, handle, protocol=pickle.HIGHEST_PROTOCOL)

    torch.cuda.empty_cache()
    return stage2est

