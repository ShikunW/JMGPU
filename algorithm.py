
# -*- coding: utf-8 -*-
"""
    algorithm.py defines all functions runs only on CPU:
    Stagquantie1(dat, ns0, lbda=np.empty(0), knots=np.empty(0)):
        calculate stage 1 estimates for variance components (Sec 2.2.1)
    calc_surv_init(dat, ns, cut_pt):
        calculate initial values for survival submodel (Sec 2.2.8)
    calc_baseline_hazard_cut_pt(surv, bandwidth):
        calculate baseline hazard cut points for survival submodel (Sec 2.1)
    calc_trapezoidal_cut_pt(surv, bandwidth):
        calculate trapezoidal rule cut points for survival submodel (Sec 2.1)
    calc_initial_value(dat, ns, beta, bandwidth):
        calculate initial values for joint model (Sec 2.2.8)
    EM_prepare(dat, stage1est, a, ns):
        calculate useful values for EM (Sec 2.2.2)
    Mstep_gamma(dBT, Egamma):
        Mstep update for gamma in EM (Sec 2.2.6)
    Mstep_beta(UUinv,UY,UV, fTdc, mu_c_pos, Sc_pos_sqrt, Z):
        Mstep update for beta in EM (Sec 2.2.5)
    Convergence(para_new, para_old, condition=0, tol=5e-3,verbose = False)
        decide convergence criteria and number of Monte Carlo points (Sec 2.2.9)
    B(t, cut_pt = np.arange(0, 11, 1)):
        calculate piecewise constant baseline hazard (Sec 2.1)
    alpha_iteration_number(conv_proportion, type = 'one-step'):
        calculate the number of iterations for newton-raphson for alpha
"""

# import useful only for development
import numpy as np
import pandas as pd
import torch
from scipy.linalg import block_diag
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import timeit


def Stage1(dat, ns0, lbda=np.empty(0), knots=np.empty(0)):
    """Stage 1 estiamtes for variance components (Sec 2.2.1)

    Args:
        dat: a dictionary of all data.
        ns0: a vector of the size parameters.
        lbda: a vector of prespecified penalty parameters, length = number of longitudinal biomarkers; empty if for linear model
        knots: a vector of knots for truncated polynomial basis; empty if for linear model

    Returns:
        A list of consistent intial value for beta, consistent variance components for Sigma_b, sigma_e.
    
    Raises:
        ValueError: the estimated covariance matrix Sigma_b is not positive definite
    """
    n, nX, nY, nt, nMC, batch_size = ns0
    deg = nt-1
    q = len(knots)
    if q == 0: # linear model
        lbda = np.zeros(nY)
    Y = dat['Y']; X = dat['X']; nn = dat['nn']; time = dat['time']
    sigma2_e = np.zeros(nY)
    beta = np.zeros((nX + nt, nY))
    bj = [np.zeros((nt, n))] * nY
    DTD = [np.zeros((nt, nt))] * nY
    intX = np.column_stack((np.ones(n), X))
    XTX = np.add.reduce([np.outer(x, x) for x in intX])
    XTXinv = np.linalg.inv(XTX)
    eta = np.zeros((nt * nY, n))
    Q = np.diag(np.concatenate((np.zeros(nt), np.ones(q))))
    C = np.column_stack((np.eye(nt), np.zeros((nt, q))))
    #Yhat = [[[]] * nY for _ in range(n)]
    for j in range(nY):
        sigma2_up = sigma2_low = 0
        XTetaj0 = np.zeros(1 + nX)
        etaj = np.zeros((nt, n))
        DTDj_inv = np.zeros((nt, nt))
        for i in range(n): # for loops can be sped up
            if q > 0:
                Dji = np.column_stack((PolyBasis(time[i][j], deg), #np.ones(nn[i][j]), time[i][j]
                                       TruncPolyBasis(time[i][j], knots, deg)))
            else:
                Dji = PolyBasis(time[i][j], deg)

            Yji = Y[i][j][:, None]
            DTDji = np.dot(Dji.T, Dji)
            DTDji_inv = np.linalg.inv(DTDji + lbda[j] * Q)
            CinvD = C.dot(DTDji_inv).dot(Dji.T)
            DTDj_inv += CinvD.dot(CinvD.T)
            Sji = Dji.dot(DTDji_inv).dot(Dji.T)
            etaji = DTDji_inv.dot(Dji.T).dot(Yji)[:nt]
            etaj[:, i] = etaji.flatten()
            XTetaj0 += intX[i] * etaji[0, 0]
            sigma2_up += Yji.T.dot(np.eye(nn[i, j]) - Sji).dot(Yji)[0, 0]
            if q > 0:
                sigma2_low += nn[i][j] - np.trace(Sji)
            else:
                sigma2_low += nn[i][j] - nt
            #Yhat[i][j] = np.dot(Sji, Yji).flatten()
        betaj0 = np.dot(XTXinv, XTetaj0)  # equation (6)
        betaj1 = np.sum(etaj[1:], axis = 1) / n  # equation (6)

        betaj = np.append(betaj0, betaj1)
        beta[:, j] = betaj
        sigma2_e[j] = sigma2_up / sigma2_low  # equation (5)
        DTD[j] = DTDj_inv * sigma2_e[j]  # Kronecker sum np.add.outer(a, b)
        bj[j] = np.column_stack([etaj[:, i] - np.append(np.dot(intX[i], betaj0), betaj1)
                                 for i in range(n)])  # equation (7)
        eta[range(j * nt, j * nt + nt), :] = etaj

    sigma_u = np.zeros(nY)
    if q > 0:
        sigma_u = np.sqrt(sigma2_e / lbda)
    if nt == 3:
        b = np.row_stack(bj)[np.array([0, 1, 3, 4])].T
    else:
        b = np.row_stack(bj).T
    Eb = np.add.reduce(b) / n
    EbbT = sum(np.outer(x, x) for x in b) / n
    DTD = np.array([x[:2,:2] for x in DTD])
    Sigma_b = EbbT - np.outer(Eb, Eb) - block_diag(*DTD) / n
    #print(np.linalg.eigvals(Sigma_b))
    if ~np.all(np.linalg.eigvals(Sigma_b) >= 0):
        print('Sigma_b=',Sigma_b)
        raise ValueError('The estimated covariance matrix Sigma_b is not positive definite!')
    stage1est = {'beta': beta, 'sigma_e': np.sqrt(sigma2_e), 'Sigma_b': Sigma_b,
                 'sigma_u': sigma_u#, 'Yhat': Yhat
                 }
    return stage1est


def calc_surv_init(dat, ns, cut_pt):
    """Calculate initial values for survival submodel (Sec 2.2.8)

    Args:
        dat: a dictionary of all data.
        ns: a vector of the size parameters.
        cut_pt: a vector of cut points for piecewise constant baseline hazard.

    Returns:
        Two vectors for alpha and gamma, which are parameters for survival submodel coefficients and baseline hazard.
    
    Raises:
    """
    # cox model for initial values of alpha and zeta
    _, nX, nY, _, _, _ = ns
    data = {'surv': dat['surv'], 'status': dat['status']}
    for i in range(nX):
        data['X' + str(i+1)] = dat['X'][:, i]
    df = pd.DataFrame(data)
    cph = CoxPHFitter()  ## Instantiate the class to create a cph object
    cph.fit(df, 'surv', event_col='status')  ## Fit the data to train the model
    alpha1 = np.array(cph.summary.coef)  # ss['alpha']
    alpha2 = np.zeros(nY)
    alpha = np.append(alpha1, alpha2)
    ch = cph.baseline_cumulative_hazard_  # baseline hazard
    ch = pd.concat([pd.DataFrame({'baseline cumulative hazard': 0}, index=[cut_pt[0] - 1e-10]), ch[:]])
    time = ch.index.values  # time where baseline hazard takes values
    # transform nonparametric estimate of baseline hazard to parametric estimate using piecewise constant assumption
    gamma = np.asarray(
        [(np.min(ch[time >= cut_pt[k + 1]]) - np.max(ch[time < cut_pt[k]])).values / (cut_pt[k + 1] - cut_pt[k])
         for k in range(len(cut_pt) - 1)])  # equation (15)
    alpha = np.nan_to_num(alpha)
    gamma = np.nan_to_num(gamma.flatten())
    return alpha, gamma


def calc_baseline_hazard_cut_pt(surv, bandwidth=1):
    """Calculate baseline hazard cut points for survival submodel (Sec 2.1)

    Args:
        surv: a vector of survival time.
        bandwidth: >-1, if bandwidth>0, fix length piecewise constant;
                f -1<bandwidth<0, dynamic piecewise constant depending on the percentile of the sample size.

    Returns:
        A vector of cut points for baseline hazard, from min(surv) to max(surv).

    Raises:
        ValueError: If bandwidth <= -1.
    """
    if bandwidth > 0:
        cut_pt = np.arange(0,1+ bandwidth / 2, bandwidth)
    elif -1 < bandwidth < 0:
        cut_pt = np.quantile(surv, q=np.arange(0, 1 - bandwidth / 2, -bandwidth))
    else:
        raise ValueError('bandwidth for baseline hazard must > -1!')
    return cut_pt


def calc_trapezoidal_cut_pt(surv, bandwidth=1):
    """Calculate trapezoidal rule cut points for survival submodel (Sec 2.1)

    Args:
        surv: a vector of survival time.
        bandwidth: >0, bandwidth of fixed interval from min to max.

    Returns:
        A vector of cut points for tranpezoidal rule, from min(surv) to max(surv).

    Raises:
        ValueError: If bandwidth <= 0.
    """
    if bandwidth > 0:
        surv_max = 1#np.max(surv)
        surv_min = 0
        cut_pt = np.arange(surv_min, surv_max + bandwidth / 2, bandwidth)
        cut_pt = np.insert(np.insert(cut_pt, len(cut_pt), surv_max), 0, surv_min)
    else:
        raise ValueError('bandwidth for tranpezoidal rule must be positive!')
    return cut_pt


def calc_initial_value(dat, ns, beta, bandwidth=1):
    """Calculate initial values for joint model (Sec 2.2.8)

    Args:
        dat: a dictionary of all data.
        ns: a vector of the size parameters.
        beta: a vector of beta from stage 1 estimate.
        bandwidth: >-1, bandwidth for baseline hazard.

    Returns:
        initial values for alpha, beta, gamma; cut_pt for baseline hazard.

    Raises:
    """

    # calculate initial value
    cut_pt = calc_baseline_hazard_cut_pt(dat['surv'], bandwidth=bandwidth)
    betaflat = np.nan_to_num(beta.T.flatten())
    alpha, gamma = calc_surv_init(dat, ns, cut_pt=cut_pt)
    return betaflat, alpha, gamma, cut_pt


def EM_prepare(dat, stage1est, a, ns0, knots=np.empty(0)):  # maybe need to accelerate for-loops
    """Calculate useful values for EM (Sec 2.2.2)

    Args:
        dat: a dictionary of all data.
        stage1est: a list of parameters from stage 1 estimate.
        beta: a vector of beta from stage 1 estimate.
        a: a vector of baseline hazard cut points.
        ns0: a vector of the size parameters.
        knots: a vector of knots for nonlinear model.

    Returns:
        a dict of values useful for EM.

    Raises:

    """
    n, nX, nY, nt, nMC, _ = ns0
    deg = nt-1
    q = len(knots)
    Y0 = dat['Y']; X = dat['X']; surv = dat['surv']; nn = dat['nn']; status = dat['status']; time = dat['time']
    sigma_e = stage1est['sigma_e']
    Sigma_b = stage1est['Sigma_b']
    sigma_u = stage1est['sigma_u']
    Y = np.array(list(map(lambda x: np.concatenate(x, axis=0)[:, None], Y0)))  # column vector of responses
    sigma_e_sq = sigma_e ** 2
    if q > 0:
        V = np.array(list(map(
            lambda y: block_diag(*list(map(
                lambda x: np.column_stack((PolyBasis(x, 1), TruncPolyBasis(x, knots, 1))),
                y))),
            time)))
    else:
        V = np.array(list(map(
            lambda y: block_diag(*list(map(
                lambda x: PolyBasis(x, 1),
                y))),
            time)))

    oneX = np.column_stack((np.ones(n), X))
    U = np.array([block_diag(*[np.column_stack((np.kron(np.ones((nn[i][j], 1), dtype=int),
                                                        oneX[i]),  # intX = [1,X1,X2]
                                                PolyBasis(time[i][j], deg, False)))
                               for j in range(nY)])
                  for i in range(n)])

    UUinv = np.linalg.inv(np.sum(list(map(lambda x: np.inner(x.T, x.T), U)), axis=0))
    UY = np.add.reduce(list(map(lambda x, y: x.T.dot(y), U, Y))).flatten()
    UV = np.array(list(map(lambda x, y: x.T.dot(y), U, V)))
    dX = status[:, np.newaxis] * X
    dXsum = np.sum(dX, axis=0)
    Sigma_c = Sigma_b
    if q > 0:
        for j in range(nY):
            Sigma_c = np.insert(Sigma_c, np.repeat(2 * (j + 1) + q * j, q), 0, axis=0)
            Sigma_c = np.insert(Sigma_c, np.repeat(2 * (j + 1) + q * j, q), 0, axis=1)
            idxj = np.arange(2 * (j + 1) + q * j, (2 + q) * (j + 1))
            Sigma_c[idxj[:, None], idxj[None, :]] = np.diag(np.repeat(sigma_u[j] ** 2, q))
    Sigma_cinv = np.linalg.inv(Sigma_c)
    sigma_e2_inv = 1 / sigma_e ** 2
    ScXY = np.array(list(map(
        lambda x, y: np.linalg.inv(x.T.dot(np.diag(np.repeat(sigma_e2_inv, y))).dot(x) +
                                    Sigma_cinv), V, nn)))
    ScXY_sqrt = np.array(list(map(lambda x: np.linalg.cholesky(x), ScXY)))

    SinvDSY = np.array(list(map(
        lambda x, y, z, w: z.dot(x.T.dot(np.diag(np.repeat(sigma_e2_inv, y))).dot(w)),
        V, nn, ScXY, Y)))
    SinvDSU = np.array(list(map(
        lambda x, y, z, w: z.dot(x.T.dot(np.diag(np.repeat(sigma_e2_inv, y))).dot(w)),
        V, nn, ScXY, U)))
    d = calc_trapezoidal_cut_pt(surv, max((.02, max(surv) / 50)))
    J = len(d) - 2
    BT = B(surv, a).T
    dBTsum = np.add.reduce(status[:, np.newaxis] * BT)  # / bandwidth
    dd = np.array([[(min(x, d[j + 1]) - d[j - 1]) * (x >= d[j - 1]) / 2
                    for j in range(1, J + 1)]
                   for x in surv])
    minTd = np.array(list(map(lambda x: list(map(lambda y: min(y, x), d[1:-1])), surv)))

    Bt = np.array([dd[:, j - 1] * B(minTd[:, j], a) for j in range(J)])
    Bt = np.moveaxis(Bt, -1, 0)

    dit = np.array([dd[:, j] * B(minTd[:, j], a) for j in range(J)])
    dit = np.moveaxis(dit, -1, 0)

    Va = np.array(list(map(
        lambda x: block_diag(*np.tile((1, x),
                                      (nY, 1, 1))), a[:-1])))
    Vt = np.array(list(map(
        lambda x: list(map(
            lambda y: block_diag(*np.tile((1, y),
                                          (nY, 1, 1))), x)), minTd)))

    if q > 0:
        Vt2 = np.array(list(map(
            lambda x: list(map(
                lambda y: block_diag(*np.tile(TruncPolyBasis(y, knots, deg),
                                              (nY, 1, 1))), x)), minTd)))
    else:
        Vt2 = np.zeros((n, 1, 1))

    VT = np.array(list(map(
        lambda x: block_diag(*np.tile((1, x),
                                      (nY, 1, 1))), surv)))
    dVT = np.array(list(map(
        lambda x, y: block_diag(*np.tile((x, y * x),
                                         (nY, 1, 1))), status, surv)))

    if q > 0:
        dVT2 = np.array(list(map(
            lambda x, y: block_diag(*np.tile(TruncPolyBasis(y, knots, deg) * x,
                                             (nY, 1, 1))), status, surv)))
    else:
        dVT2 = np.zeros((n, 1))

    # prepare of components for likelihood calculation
    Sigma = np.array(list(map(lambda x:
                np.diag(np.concatenate(list(map(lambda y, z:
                    np.repeat(y, z), sigma_e_sq, x)))), nn)))

    Vpos = np.array(list(map(lambda si, vi: si + vi.dot(Sigma_c).dot(vi.T), Sigma, V)))
    Vpos_inv = np.array(list(map(lambda x: np.linalg.inv(x), Vpos)))

    tmp1 = np.sum(nn) * np.log(2 * np.pi)
    tmp2 = np.array(list(map(lambda x: np.clip(np.log(np.linalg.det(x)),-1e4,1e4), Vpos)))
    VlogLik = -.5 * (tmp1 + np.sum(tmp2))

    return UUinv, UY, UV, dX, dXsum, dVT, VT, BT, dBTsum, \
           ScXY_sqrt, SinvDSY, SinvDSU, d, dit, Bt, Vt, Va, Vt2, dVT2, \
           Y, U, VlogLik, Vpos_inv


def Estep_gen_c_pos(beta_old, SinvDSY, SinvDSU, ns):
    """get posterior mean of random effects c and a random matrix from Normal(0,1)

    Args:
        beta_old: a vector of longitudinal model fixed effects beta at previous iteration
        SinvDSY: pre-calculated matrix, useful for posterior mean of random effects c
        SinvDSU: pre-calculated matrix, useful for posterior mean of random effects c
        ns: a vector of the size parameters.

    Returns:
        Z, ~ Normal(0,1) for sampling random effects c, dim = nMC*nMC.

    Raises:

    """
    nMC = ns[4]
    mu_c_pos = np.array(SinvDSY - SinvDSU.dot(beta_old.T[:, np.newaxis]))
    mu_c_pos = mu_c_pos.squeeze(2)
    Zlen = mu_c_pos.shape[1]
    Z = np.random.multivariate_normal(np.zeros(Zlen), np.eye(Zlen), nMC)
    return mu_c_pos, Z


def Mstep_gamma(dBT, Egamma):
    """Mstep update for gamma in EM (Sec 2.2.6)

    Args:
        dBT: pre-calculated sum(delta * I(T in Ij)), j = 1,...,len(a), dim = J*1.
        Egamma: pre-calculated sum(U'Y), dim = J*1.

    Returns:
        new gamma.

    Raises:

    """
    gamma_new = dBT / Egamma  # equation (13)
    return gamma_new


def Mstep_beta(UUinv, UY, UV_tensor, fTdc_tensor, mu_c_pos_tensor, Sc_pos_sqrt_tensor, Z_tensor):
    """Mstep update for beta in EM (Sec 2.2.5)

    Args:
        UUinv: pre-calculated sum(U'U)^{-1}, dim = (nX+2)*(nX+2).
        UY: pre-calculated sum(U'Y), dim = (nX+2)*1.
        UV: pre-calculated sum(U'V)^{-1}, dim = (nX+2)*2.
        fTdc: f(c|Y,T,delta;theta) at current step, dim = n*nMC*(2nY).
        mu_c_pos, posterior mean of random effects c, dim = n*(2nY).
        Sc_pos_sqrt, posterior variance of random effects c, dim = n*(2nY)*(2nY).
        Z, ~ Normal(0,1) for sampling random effects c, dim = nMC*nMC.

    Returns:
        new beta.

    Raises:

    """
    UDb = np.add.reduce(
        list(map(lambda x, y, z, w: (torch.matmul(x, ((torch.matmul(y, Z_tensor.T) + z[:, np.newaxis]) *
                                                      w[np.newaxis, :])).sum(axis=1)).cpu().numpy(),
                 UV_tensor, Sc_pos_sqrt_tensor, mu_c_pos_tensor, fTdc_tensor)))
    # numerator / denominator in equation (15)

    beta_new = UUinv.dot(UY - UDb)  # equation (12)
    return beta_new


def Convergence(history, verbose=False, control ={'tol': 5e-3, 'condition': 'rel_diff', 'MCscaler': 2,
                          'MCmax': 2e5, 'EMmax': 200, 'trace_back': 3, 'test': False}):
    """Decide convergence criteria and number of Monte Carlo points (Sec 2.2.9)

    Args:
        para_new: a vector of new parameters at current EM iteration.
        para_old: a vector of old parameters at previous EM iteration.
        logLik_new: new log-likelihood at current EM iteration.
        logLik_old: old log-likelihood at previous EM iteration.
        verbose: =True, print convergence details
        control: a dict of convergence control parameter.
            tol: tolerance parameter, general EM uses 1e-6; joineRML uses 5e-3.
            condition: ="rel_diff", relative difference |para_new - para_old| / |para_old| < tol.
                       ="abs_diff", absolute difference |para_new - para_old| < tol.
            verbose: =1, print number of converged parameters.
            MCscaler: a small scaler to increase MC <- MC + MC / MCscaler when CV condition is satisfied.
            MCmax: maximum allowed number of Monte Carlo points, should depend on the memory of device.
            EMmax: maximum allowed number of EM iterations.

    Returns:
        cond[condition]: True if satisfy convergence criteria.
        delta_tmp: coefficient of variation to decide whether to increase number of MC points.
        tmp00: a vector of indicator, =1 if the corresponding parameter satisfy convergence criteria.

    Raises:

        """
    para_new = history['param'][-1]; para_old = history['param'][-2]
    logLik_new = history['logLik'][-1]; logLik_old = history['logLik'][-2]
    tol1 = 1e-6
    count = len(history['param'][0])
    # Absolute parameter change
    abs_diff = np.absolute(para_new - para_old)
    abs_ind = 1 * (abs_diff < control['tol'])
    history['param_abs_ind'].append(abs_ind)
    # Relative parameter change
    #..\aten\src\ATen\native\BinaryOps.cpp: 81: UserWarning: Integer division of tensors using div or / is deprecated,
    # and in a future release div will perform true division as in Python 3. Use true_divide or floor_divide( // in Python) instead.
    rel_diff = abs_diff / (np.absolute(para_old) + tol1)
    rel_ind = 1 * (rel_diff < control['tol'])
    history['param_rel_ind'].append(rel_ind)

    # Relative logLik change
    det_absConv = np.min(history['param_abs_ind'][-control['trace_back']:]) == 1
    det_relConv = np.min(history['param_rel_ind'][-control['trace_back']:]) == 1

    delta_rel = rel_diff.max()
    rel_diff_logLik = np.absolute(logLik_new - logLik_old) / (np.absolute(logLik_old) + tol1)
    det_rel_logLik = rel_diff_logLik < tol1
    if verbose:
        print('Converged number of parameter; total=', count, "; tol=", control['tol'],
              '\nabsConv=', abs_ind.sum(), ";", det_absConv, ";", 1*history['param_abs_ind'][-1],
              '\nrelConv=', rel_ind.sum(), ";", det_relConv, ";", 1*history['param_rel_ind'][-1],
              '\nlogLik= '+str(np.round(logLik_new, 1)) + " vs " + str(np.round(logLik_old, 1)) +
              " (" + str(det_rel_logLik) + ")",
              "\nnumber of MC points=", history['nMC'][-1])
    if control['condition'] == 'rel_diff':
        conv = det_relConv
        conv_proportion = abs_ind.sum() / count
    else:
        conv = det_absConv
        conv_proportion = rel_ind.sum() / count

    history['logLik_rel_ind'].append(det_rel_logLik*1)
    if len(history['param']) >= control['EMmax']:
        conv = True
        if verbose:
            print('Warning: reach maximum allowed EM iteration, you need to increase the EMmax variable in control setting')
    return conv, delta_rel, conv_proportion, history


def B(t, cut_pt=np.arange(0, 1.1, .5)):  # piecewise constant baseline hazard
    """Calculate piecewise constant baseline hazard (Sec 2.1)

    Args:
        t: a value or vector of time within [0, max(surv)].
        cur_pt: a vector of cut point for piecewise constant baseline hazard, dim = J.

    Returns:
        Bt: B(t)=[1(t in I1),...,1(t in IJ)].
        """
    cut_pt = np.unique(cut_pt)
    p = len(cut_pt) - 1
    pt = len(t)
    Bt = np.zeros((pt, p))
    cut_pt[p] = 999
    coords = np.column_stack((range(pt), np.searchsorted(cut_pt, t, side='right') - 1))
    Bt[tuple(coords.T)] = 1  # decide which interval t is in
    return Bt.T


def alpha_iteration_number(conv_proportion, type='one-step'):
    """Calculate the number of iterations for newton-raphson for alpha

    Args:
        conv_proportion: a value or vector of time within [0, max(surv)].
        type: ='one-step', return 1; ='dynamic', depend on the proportion of converged. parameters, the more parameter converged, the more interation is allowed to run.

    Returns:
        a number as allowed maximum iteration for Estep of alpha.
    """
    if type == 'one-step':
        return 1
    elif type == 'dynamic':
        return int(conv_proportion * 5) + 1


def TruncPolyBasis(t, knots=np.array([0]), deg = 1):
    q = len(knots)
    if type(t) is not np.ndarray:
        t = np.array([t])
    p = t.shape[0]
    if knots[0] == 999:
        out = np.sin(t * np.pi)
    else:
        t0 = np.zeros((p, q))
        t1 = np.kron(np.ones((1, q)), t[:, np.newaxis])
        t2 = np.kron(knots, np.ones((p, 1)))
        out = np.maximum(t0, t1 - t2) ** deg
    return out


def PolyBasis(t, deg = 1, intercept = True):
    if type(t) is not np.ndarray:
        t = np.array([t])
    if deg == 0 and intercept == True:
        out = np.ones(t.shape[0])
    elif deg == 1 and intercept == True:
        out = np.column_stack((np.ones(t.shape[0]), t))
    elif deg == 2 and intercept == True:
        out = np.column_stack((np.ones(t.shape[0]), t, t ** 2))
    if deg == 0 and intercept == False:
        out = np.zeros(t.shape[0])
    elif deg == 1 and intercept == False:
        out = t
    elif deg == 2 and intercept == False:
        out = np.column_stack((t, t ** 2))
    return out


def select_lbda(dat, ns, knots):
    n, nX, nY, nt, nMC, batch_size = ns
    deg = nt-1
    Y = dat['Y']; X = dat['X']; nn = dat['nn']; time = dat['time']
    sigma2_e = np.zeros(nY)
    beta = np.zeros((nX + nt, nY))
    bj = [np.zeros((nt, n))] * nY
    DTD = [np.zeros((nt, nt))] * nY
    oneX = np.column_stack((np.ones(n), X))
    XTX = np.add.reduce(list(map(lambda x: np.outer(x, x), oneX)))
    XTXinv = np.linalg.inv(XTX)
    eta = np.zeros((nt * nY, n))
    q = len(knots)
    Q = np.diag(np.concatenate((np.zeros(nt), np.ones(q))))
    C = np.column_stack((np.eye(nt), np.zeros((nt, q))))
    lbda_lst0 = range(-2, 7)
    lbda_lst = [10 ** x for x in lbda_lst0]
    gcv = np.zeros((len(lbda_lst), nY))

    # select lbda
    for k in range(len(lbda_lst)):
        lbda = lbda_lst[k]
        for j in range(nY):
            gcv_up = 0
            gcv_low = 0
            sigma2_up = sigma2_low = 0
            XTetaj0 = np.zeros(1 + nX)
            etaj = np.zeros((nt, n))
            DTDj_inv = np.zeros((nt, nt))
            for i in range(n):
                Dji = np.column_stack((PolyBasis(time[i][j], deg), TruncPolyBasis(time[i][j], knots, deg)))
                Yji = Y[i][j][:, None]
                DTDji = np.dot(Dji.T, Dji)
                DTDji_inv = np.linalg.inv(DTDji + lbda * Q)
                CinvD = C.dot(DTDji_inv).dot(Dji.T)
                DTDj_inv += CinvD.dot(CinvD.T)
                Sji = Dji.dot(DTDji_inv).dot(Dji.T)
                etaji = DTDji_inv.dot(Dji.T).dot(Yji)[:nt]
                etaj[:, i] = etaji.flatten()
                XTetaj0 += oneX[i] * etaji[0, 0]
                sigma2_up += Yji.T.dot(np.eye(nn[i][j]) - Sji).dot(Yji)[0, 0]
                sigma2_low += nn[i][j] - np.trace(Sji)
                gcv_up += Yji.T.dot(np.eye(nn[i][j]) - Sji).dot(Yji)[0, 0]
                gcv_low += (1 - np.trace(Sji) / nn[i][j]) ** 2
            betaj0 = XTXinv.dot(XTetaj0)  # betaj0 = ss['beta_long'][:,j]
            betaj1 = np.sum(etaj[1:], axis = 1) / n  # equation (6)

            betaj = np.append(betaj0, betaj1)
            beta[:, j] = betaj
            sigma2_e[j] = sigma2_up / sigma2_low
            gcv[k, j] = gcv_up / gcv_low
            DTD[j] = DTDj_inv * sigma2_e[j]
            bj[j] = np.column_stack([etaj[:, i] - np.append(np.dot(oneX[i], betaj0), betaj1)
                                     for i in range(n)])
            eta[range(j * nt, j * nt + nt), :] = etaj

    for j in range(nY):
        plt.plot(lbda_lst0, gcv[:, j])
        plt.xlabel('10^x')
        plt.ylabel('GCV')
        plt.title('Y' + str(j))
        # plt.show()

    lbda = [lbda_lst[x] for x in np.argmin(gcv, axis=0)]
    print("selected lbda = " + str(lbda))
    return lbda


def GOF(likelihood, n, p):
    aic = -2 * likelihood + 2 * p
    bic = -2 * likelihood + p * np.log(n)
    return aic, bic


def evaluate(ns, stage2est, llkh):
    # calculate AIC, BIC, AUC, MSE
    # ns = np.array([200, 2, 2, 2, 100, 0, 2, 51])
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns

    p = len(stage2est['beta']) + len(stage2est['alpha']) + len(stage2est['gamma']) + \
        len(stage2est['sigma_e']) + len(stage2est['Sigma_b'].flatten())
    if q > 0:
        p += len(stage2est['sigma_u'])

    aic, bic = GOF(llkh, n, p)
    # long_accuracy, surv_accuracy = Accuracy(ns, dat, stage2est, a)

    output = {'aic': aic, 'bic': bic  # , 'long_accuracy': long_accuracy, 'surv_accuracy': surv_accuracy
              }

    return output


def likelihood(VlogLik, Vpos_inv, beta, Y, U, fTdc_denom):
    """calculate the log-likelihood """
    # beta, alpha, gamma = beta_new, alpha_new, gamma_new
    residual = np.array(list(map(lambda y, u: y - np.dot(u, beta)[:, np.newaxis], Y, U)))
    rVr = np.array(list(map(lambda r, v: r.T.dot(v).dot(r)[0, 0], residual, Vpos_inv)))
    logLik_long = VlogLik - .5 * np.sum(rVr)
    logLik_surv = fTdc_denom
    #print("logLik_long=", logLik_long)
    #print("logLik_surv=", logLik_surv)
    logLik = logLik_long + logLik_surv
    return logLik


# the following code need to be checked for dynamic prediction and its evaluation

def gen_new_data(ns, beta, surv_new=.5, type='linear'):
    """generate ideal data for prediction example"""
    n, nX, nY, nt, nMC, J, K, q = ns
    # generate simulation data
    X_new = np.array([.5, 0])
    oneX_new = np.insert(X_new, 0, 1)
    time_new = np.tile(np.arange(-.04, surv_new + .0001, .02), (2, 1))
    nn_new = np.array([len(x) for x in time_new])
    time_pred = np.tile(np.arange(surv_new, 1.0001, .02), (nY, 1))

    if type == 'linear':
        Y_new = np.array(list(map(lambda x:
                                  list(map(lambda y: np.dot(beta[range(x * (2 + nX), (x + 1) * (2 + nX))],
                                                            np.append(oneX_new, y)), time_new[x])), range(nY))))
    elif type == 'nonlinear':
        u_new = np.array([1, -1])
        Wu_new = np.array([u_new[j] * np.sin(time_new[j] / 6 * np.pi) for j in range(nY)])
        Y_new = np.array(list(map(lambda x:
                                  list(map(lambda y: np.dot(beta[range(x * (2 + nX), (x + 1) * (2 + nX))],
                                                            np.append(oneX_new, time_new[x][y])) + Wu_new[x][y],
                                           range(nn_new[x]))), range(nY))))
    newdata = {'Y': Y_new, 'X': X_new, 'nn': nn_new, 'time': time_new, 'time_pred': time_pred}
    return newdata


def predict(newdata, ns, dat, stage2est, a, knots=np.array([0, .2, .4, .6, .8])):
    """predict the event probability in time interval (s, s+t1]"""

    # get fitted values
    n, nX, nY, nt, nMC, q, J, K = ns
    deg = nt-1
    alpha = stage2est['alpha']
    beta = stage2est['beta']
    gamma = stage2est['gamma']
    sigma_e = stage2est['sigma_e']
    Sigma_b = stage2est['Sigma_b']
    if q>0:
        sigma_u = stage2est['sigma_u']

    surv = dat['surv']
    sigma_e2_inv = 1 / sigma_e ** 2
    surv_new = newdata['time_pred'][0, 0]
    if q>0:
        Sigma_cinv = np.linalg.inv(block_diag(Sigma_b, np.diag(np.repeat(sigma_u, q))))
    else:
        Sigma_cinv = np.linalg.inv(Sigma_b)
    alpha0 = alpha[np.arange(nX)]
    alpha1 = alpha[np.arange(nX, alpha.shape[0])]

    d = calc_trapezoidal_cut_pt(surv, .02) # max((.02, max(surv) / 50))

    # get new data
    Y_new = newdata['Y']
    X_new = newdata['X']
    time_new = newdata['time']
    time_pred = newdata['time_pred']
    oneX_new = np.insert(X_new, 0, 1)
    nn_new = np.array([len(x) for x in time_new])
    nn_pred = np.array([len(x) for x in time_pred])

    U_new = block_diag(*[np.column_stack((np.kron(np.ones((nn_new[j], 1), dtype=int), oneX_new),  # intX = [1,X1,X2]
                                          time_new[j])) for j in range(nY)])


    if q>0:
        V_new = np.array(block_diag(
            *list(map(lambda x: np.column_stack((np.ones(len(x)), x, TruncPolyBasis(x, knots, deg))), time_new))))
    else:
        V_new = np.array(block_diag(*list(map(lambda x: np.column_stack((np.ones(len(x)), x)), time_new))))
    ScXY_new = np.linalg.inv(V_new.T.dot(np.diag(np.repeat(sigma_e2_inv, nn_new))).dot(V_new) + Sigma_cinv)
    SinvDSY_new = ScXY_new.dot(V_new.T.dot(np.diag(np.repeat(sigma_e2_inv, nn_new))).dot(Y_new.flatten()))
    SinvDSU_new = ScXY_new.dot(V_new.T.dot(np.diag(np.repeat(sigma_e2_inv, nn_new))).dot(U_new))
    mu_c_pos_new = SinvDSY_new.flatten() - SinvDSU_new.dot(beta)

    Zlen = len(mu_c_pos_new)
    Z_new = np.random.multivariate_normal(np.zeros(Zlen), np.eye(Zlen), 1000)

    Sc_pos_sqrt_new = np.linalg.cholesky(ScXY_new)

    c_new = np.array([Sc_pos_sqrt_new.dot(x) + mu_c_pos_new for x in Z_new]).T
    effect1 = np.exp(np.dot(X_new, alpha0))

    u_lst = np.arange(0, 1.0005, .0005)
    u_pred = np.insert(u_lst[u_lst > surv_new], 0, surv_new)
    prob_lst = np.empty((0, 100))
    for u in u_pred:
        minTd_new = np.array(list(map(lambda x: min(x, u), d[1:-1])))

        dd_new = np.array([(min(u, d[j + 1]) - d[j - 1]) * (d[j - 1] <= u) / 2
                           for j in range(1, K + 1)])

        Bt_new = np.array([dd_new[j - 1] * B([minTd_new[j]], a).flatten()  # for trapezoidal rule
                           for j in range(K)])

        cumbaselinehazard = np.dot(Bt_new, gamma)

        Vt_new = np.array(list(map(lambda x: block_diag(*np.tile((1, x), (nY, 1, 1))), minTd_new)))
        Gb_new = np.dot(np.ascontiguousarray(Vt_new), c_new[range(nt * nY)])


        if q > 0:
            Vt2_new = np.array(list(map(lambda x: block_diag(*np.tile(TruncPolyBasis(x, knots, deg), (nY, 1))), minTd_new)))
            Wu_new = np.dot(np.ascontiguousarray(Vt2_new), c_new[range(nt * nY, (nt + q) * nY)])
            effect2 = np.array([np.sum(
                np.exp(np.dot(np.ascontiguousarray(Gb_new[:, :, x] + Wu_new[:, :, x]), alpha1)) * cumbaselinehazard) for
                                x in range(100)])
        else:
            effect2 = np.array(
                [np.sum(np.exp(np.dot(np.ascontiguousarray(Gb_new[:, :, x]), alpha1)) * cumbaselinehazard) for x in
             range(100)])
        prob = np.array([list(map(lambda x: np.exp(-effect1 * x), effect2))])
        prob_lst = np.append(prob_lst, prob, axis=0)

    # calculate mean and confidence interval of survival prob
    prob_lst += np.ones(prob_lst.shape)*1e-10
    prob_lst /= np.tile(prob_lst[0], (len(u_pred), 1))

    prob_mean_lst = np.mean(prob_lst, axis=1)
    prob_lb_lst = np.quantile(prob_lst, 0.025, axis=1)
    prob_ub_lst = np.quantile(prob_lst, 0.975, axis=1)
    # plt.plot(u_pred,prob_mean_lst);plt.plot(u_pred,prob_lb_lst);plt.plot(u_pred,prob_ub_lst);plt.show()
    #######################

    Yhat_new_mean_lst = [[]] * nY;
    Yhat_new_lb_lst = [[]] * nY;
    Yhat_new_ub_lst = [[]] * nY
    Yhat_pred_mean_lst = [[]] * nY;
    Yhat_pred_lb_lst = [[]] * nY;
    Yhat_pred_ub_lst = [[]] * nY
    for j in range(nY):
        betaj = beta[range(j * (2 + nX), (j + 1) * (2 + nX))]
        Uji_new = np.column_stack((np.kron(np.ones((nn_new[j], 1), dtype=int), np.insert(X_new, 0, 1)), time_new[j]))
        Uji_pred = np.column_stack((np.kron(np.ones((nn_pred[j], 1), dtype=int), np.insert(X_new, 0, 1)), time_pred[j]))

        if q > 0:
            Dji_new = np.column_stack((np.ones(nn_new[j]), time_new[j], TruncPolyBasis(time_new[j], knots, deg)))
            Dji_pred = np.column_stack((np.ones(nn_pred[j]), time_pred[j], TruncPolyBasis(time_pred[j], knots, deg)))
            Yhat_new = np.tile(np.dot(Uji_new, betaj), (1000, 1)).T + np.dot(Dji_new, c_new[
                range(j * (nX + q), (j + 1) * (nX + q))])
            Yhat_pred = np.tile(np.dot(Uji_pred, betaj), (1000, 1)).T + np.dot(Dji_pred, c_new[
                range(j * (nX + q), (j + 1) * (nX + q))])
        else:
            Dji_new = np.column_stack((np.ones(nn_new[j]), time_new[j]))
            Dji_pred = np.column_stack((np.ones(nn_pred[j]), time_pred[j]))
            Yhat_new = np.tile(np.dot(Uji_new, betaj), (1000, 1)).T + np.dot(Dji_new,
                                                                             c_new[range(j * nX, (j + 1) * nX)])
            Yhat_pred = np.tile(np.dot(Uji_pred, betaj), (1000, 1)).T + np.dot(Dji_pred,
                                                                               c_new[range(j * nX, (j + 1) * nX)])
        Yhat_new_mean_lst[j] = np.mean(Yhat_new, axis=1)
        Yhat_new_lb_lst[j] = np.quantile(Yhat_new, 0.025, axis=1)
        Yhat_new_ub_lst[j] = np.quantile(Yhat_new, 0.975, axis=1)
        Yhat_pred_mean_lst[j] = np.mean(Yhat_pred, axis=1)
        Yhat_pred_lb_lst[j] = np.quantile(Yhat_pred, 0.025, axis=1)
        Yhat_pred_ub_lst[j] = np.quantile(Yhat_pred, 0.975, axis=1)

    #######################
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #######################
    lns1 = ax.plot(time_new[0], Y_new[0], 's', c='m', marker='o', label='Y1')
    lns2 = ax.plot(time_new[1], Y_new[1], 's', c='c', marker='D', label='Y2')
    lns11 = ax.plot(time_new[0], Yhat_new_mean_lst[0], c='m')
    lns21 = ax.plot(time_new[1], Yhat_new_mean_lst[1], c='c')
    lns12 = ax.plot(time_pred[0], Yhat_pred_mean_lst[0], c='m', linestyle='dotted')
    lns22 = ax.plot(time_pred[1], Yhat_pred_mean_lst[1], c='c', linestyle='dotted')
    lns0 = ax.plot([surv_new, surv_new], [-10, 10], label='censored time', c='g', linestyle='dashed')
    ax2 = ax.twinx()
    lns31 = ax2.plot(u_pred, prob_mean_lst, 'r-', label='mean')
    lns32 = ax2.plot(u_pred, prob_lb_lst, 'b-', linestyle='dashed', label='95% CI')
    lns33 = ax2.plot(u_pred, prob_ub_lst, 'b-', linestyle='dashed')
    # added these three lines
    lns = lns0 + lns1 + lns2 + lns11 + lns21 + lns12 + lns22 + lns31 + lns32 + lns33
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.grid()
    ax.set_ylabel('Biomarkers')
    ax2.set_ylabel('Survival probability')
    ax.set_xlabel('Time')
    ax.set_ylim(-3, 3)
    ax2.set_ylim(0, 1.1)
    ax.set_xlim(-0.05, 1.001)
    plt.title('Simulated subject prediction results at Year = ' + str(surv_new))
    plt.show()
    #######################

    out = {'newdata': newdata, 'time_pred': time_pred,
           'Y_pred': Yhat_pred_mean_lst, 'Y_pred_95ub': Yhat_pred_ub_lst, 'Y_pred_95lb': Yhat_pred_lb_lst,
           'Y_pred': Yhat_pred_mean_lst, 'Y_pred_95ub': Yhat_pred_ub_lst, 'Y_pred_95lb': Yhat_pred_lb_lst,
           'surv_prob': prob_mean_lst, 'surv_prob_95lb': prob_lb_lst, 'surv_prob_ub': prob_ub_lst}
    return (out)


def surv_accuracy(dat, stage2est, plot_ROC = False):
    """calculate time-dependent ROC"""
    # get fitted values
    ns = stage2est['ns']
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns
    deg = nt-1
    alpha = stage2est['alpha']
    beta = stage2est['beta']
    gamma = stage2est['gamma']
    sigma_e = stage2est['sigma_e']
    Sigma_b = stage2est['Sigma_b']
    sigma_u = stage2est['sigma_u']
    knots = stage2est['knots']
    a = stage2est['a']
    surv = dat['surv']
    status = dat['status']
    time = dat['time']
    X = dat['X']
    Y = dat['Y']

    nt2 = min(nt, 2)
    idx = np.arange((nt2 + q) * nY).reshape((nY, nt2 + q))
    idx1 = idx[:,np.arange(nt2)].flatten()
    idx2 = idx[:,np.arange(nt2, nt2+q)].flatten()
    # Y = np.array(list(map(lambda x: np.concatenate(x, axis=0)[:, None], Y)))
    oneX = np.column_stack((np.ones(n), X))

    sigma_e2_inv = 1 / sigma_e ** 2
    # Sigma_binv = np.linalg.inv(block_diag(Sigma_b, np.diag(np.repeat(sigma_u, q))))

    Sigma_c = Sigma_b
    if q > 0:
        for j in range(nY):
            Sigma_c = np.insert(Sigma_c, np.repeat(2 * (j + 1) + q * j, q), 0, axis=0)
            Sigma_c = np.insert(Sigma_c, np.repeat(2 * (j + 1) + q * j, q), 0, axis=1)
            idxj = np.arange(2 * (j + 1) + q * j, (2 + q) * (j + 1))
            Sigma_c[idxj[:, None], idxj[None, :]] = np.diag(np.repeat(sigma_u[j] ** 2, q))
    Sigma_cinv = np.linalg.inv(Sigma_c)

    alpha0 = alpha[np.arange(nX)]
    alpha1 = alpha[np.arange(nX, alpha.shape[0])]
    d = calc_trapezoidal_cut_pt(surv, .02) # max((.02, max(surv) / 50))
    surv_unique = np.unique(surv)

    t1_lst = np.array([.1, .3])  # time horizon
    s_lst = np.arange(.3, .8, .2)  # prediction horizon
    c_lst = np.arange(0, 1, .01)
    auc = np.zeros(len(s_lst) * len(t1_lst))
    brier_score = np.zeros(len(s_lst) * len(t1_lst))
    Se = np.zeros((len(s_lst) * len(t1_lst), len(c_lst))) * np.nan;
    Sp = np.zeros((len(s_lst) * len(t1_lst), len(c_lst))) * np.nan
    for k1 in range(len(t1_lst)):
        for k2 in range(len(s_lst)):
            t1 = t1_lst[k1]
            s = s_lst[k2]
            k = k1 * len(s_lst) + k2
            Rs = np.where(surv > s)[0]
            nk = len(Rs)
            time_new = np.array([[time[i][j][np.where(time[i][j] <= s)[0]]
                                  for j in range(nY)] for i in range(n)])
            nn_new = np.array([[len(time_new[i][j])
                                for j in range(nY)] for i in range(n)])
            Y_new = np.array([np.concatenate([Y[i][j][range(nn_new[i][j])]
                                              for j in range(nY)], axis=0)[:, None] for i in range(n)])
            probk = np.zeros(nk)
            delta_tilde = np.zeros(nk)

            for idx in range(nk):
                i = Rs[idx]
                minTd_new = np.array(list(map(lambda x: min(x, t1 + s), d[1:-1])))
                dd_new = np.array([(min(t1 + s, d[j + 1]) - d[j - 1]) * (s < d[j - 1] <= t1 + s) / 2
                                   for j in range(1, K + 1)])
                Bt_new = np.array([dd_new[j - 1] * B([minTd_new[j]], a)  # for trapezoidal rule
                                   for j in range(K)])

                Bt_new = np.moveaxis(Bt_new, -1, 0)

                cumbaselinehazard = np.dot(Bt_new, gamma)

                U_new = block_diag(
                    *[np.column_stack((np.kron(np.ones((nn_new[i][j], 1), dtype=int), oneX[i]),  # intX = [1,X1,X2]
                                       PolyBasis(time_new[i][j], deg, False))) for j in range(nY)])

                if q > 0:
                    V_new = np.array(block_diag(*list(map(lambda x: np.column_stack((PolyBasis(x, 1),
                                                                                     TruncPolyBasis(x, knots, deg))),
                                                          time_new[i]))))
                else:
                    V_new = np.array(block_diag(*list(map(lambda x: PolyBasis(x, 1), time_new[i]))))

                ScXY_new = np.linalg.inv(
                    V_new.T.dot(np.diag(np.repeat(sigma_e2_inv, nn_new[i]))).dot(V_new) + Sigma_cinv)

                SinvDSY_new = ScXY_new.dot(V_new.T.dot(np.diag(np.repeat(sigma_e2_inv, nn_new[i]))).dot(Y_new[i]))
                SinvDSU_new = ScXY_new.dot(V_new.T.dot(np.diag(np.repeat(sigma_e2_inv, nn_new[i]))).dot(U_new))
                mu_c_pos_new = SinvDSY_new.flatten() - SinvDSU_new.dot(beta)

                Vt_new = np.array(list(map(lambda x: block_diag(*np.tile((1, x), (nY, 1, 1))), minTd_new)))
                Gb_new = np.dot(np.ascontiguousarray(Vt_new), mu_c_pos_new[idx1])

                effect1 = np.exp(np.dot(X[i], alpha0))

                if q > 0:
                    Vt2_new = np.array(
                        list(map(lambda x: block_diag(*np.tile(TruncPolyBasis(x, knots, deg), (nY, 1))), minTd_new)))
                    Wu_new = np.dot(np.ascontiguousarray(Vt2_new), mu_c_pos_new[idx2])
                    effect2 = np.sum(np.exp(np.dot(np.ascontiguousarray(Gb_new + Wu_new), alpha1)) * cumbaselinehazard)
                else:
                    effect2 = np.sum(np.exp(np.dot(np.ascontiguousarray(Gb_new), alpha1)) * cumbaselinehazard)

                probk[idx] = 1 - np.exp(-effect1 * effect2)
                delta_tilde[idx] = status[i] * (s < surv[i] <= s + t1)

            h = 1  # bandwidth
            Edelta_tilde = delta_tilde

            '''
            for idx in np.where(delta_tilde == 0)[0]:
                i = Rs[idx]
                S1_hat = 1.
                S2_hat = 1.
                for ksi in surv_unique[surv_unique <= s + t1]:
                    numerator = np.sum(
                        [kernel(probk[i], probk[j], h) * (surv[j] == ksi) * delta_tilde[j] for j in range(nk)])
                    denominator = np.sum([kernel(probk[i], probk[j], h) * (surv[j] >= ksi) for j in range(nk)])
                    S1_hat *= 1 - numerator / denominator

                for ksi in surv_unique[surv_unique <= Y[i]]:
                    numerator = np.sum(
                        [kernel(probk[i], probk[j], h) * (surv[j] == ksi) * delta_tilde[j] for j in range(nk)])
                    denominator = np.sum([kernel(probk[i], probk[j], h) * (surv[j] >= ksi) for j in range(nk)])
                    S2_hat *= 1 - numerator / denominator
                Edelta_tilde[idx] = (1 - (1 - delta_tilde[idx]) * S1_hat / S2_hat)        
            '''

            # calculate AUC
            auck_numerator = np.sum([[Edelta_tilde[i] * (1 - Edelta_tilde[j]) * (1 * (probk[i] > probk[j]) +
                                                                                 .5 * (probk[i] == probk[j])) for j in
                                      range(nk)] for i in range(nk)])
            auck_denominator = np.sum([[Edelta_tilde[i] * (1 - Edelta_tilde[j]) for j in range(nk)] for i in range(nk)])
            auc[k] = auck_numerator / auck_denominator

            for j in range(len(c_lst)):
                Se[k, j] = np.sum([Edelta_tilde[i] * (probk[i] > c_lst[j]) for i in range(nk)]) / np.sum(
                    [Edelta_tilde[i] for i in range(nk)])
                Sp[k, j] = np.sum([(1 - Edelta_tilde[i]) * (probk[i] <= c_lst[j]) for i in range(nk)]) / np.sum(
                    [(1 - Edelta_tilde[i]) for i in range(nk)])

            # calculate Brier Score
            brier_score[k] = np.mean(Edelta_tilde * (1 - probk) ** 2 + (1 - Edelta_tilde) * (0 - probk) ** 2)
    if plot_ROC:
        for k in range(len(s_lst)):
            # calculate AUC
            r = np.column_stack((np.flip(1 - Sp[k]), np.flip(Se[k])))
            r = np.insert(r, 0, [0, 0], axis=0)

            # This AUC approximation may be less accurate
            # for l in range(1, len(Se2[:, k])):
            #    auc[k] += (r[l, 1] + r[l - 1, 1]) * (r[l, 0] - r[l - 1, 0]) / 2

            plt.plot(r[:, 0], r[:, 1], c=(k == 0) * 'b' + (k == 1) * 'g' + (k == 2) * 'r',
                     label='ROC at time ' + str(s_lst[k].round(2)) + ' (AUC=' + str(np.round(auc[k], 2)) + ')')
        plt.plot([0, 1], [0, 1], c='grey', linestyle='dashed')
        plt.title('ROC Curve at time .3,.5,.7')
        plt.legend()
        plt.xlabel('1-specifity')
        plt.ylabel('sensitivity')
        plt.show()
    #AUC = np.mean(auc)

    return auc, brier_score, Se, Sp


def kernel(x, y, h):
    """Gaussian kernel function"""
    u = (x-y)/h
    return(np.exp(-.5*u**2))


def long_accuracy(dat, stage2est):
    ns = stage2est['ns']
    n, nX, nY, nt, nMC, batch_size, q, J, K = ns
    deg = nt - 1
    N = np.sum(dat['nn'])
    nt2 = min(nt, 2)
    idx = np.arange((nt2 + q) * nY).reshape((nY, nt2 + q))
    idx1 = idx[:, np.arange(nt2)].flatten()
    idx2 = idx[:, np.arange(nt2, nt2 + q)].flatten()

    nn = dat['nn']
    surv = dat['surv']
    time = dat['time']
    X = dat['X']
    #Y = np.array(list(map(lambda x: np.concatenate(x, axis=0), dat['Y'])))  # column vector of responses
    Y = dat['Y']
    knots = stage2est['knots']
    Sigma_b = stage2est['Sigma_b']
    sigma_u = stage2est['sigma_u']
    sigma_e = stage2est['sigma_e']
    beta = stage2est['beta']
    beta1 = beta.reshape((nY, nt + nX))
    t1_lst = np.array([.1,.3])  # time horizon
    s_lst = np.arange(.3, .8, .2)  # prediction horizon
    mse = np.zeros(len(s_lst) * len(t1_lst))
    mae = np.zeros(len(s_lst) * len(t1_lst))
    for k1 in range(len(t1_lst)):
        for k2 in range(len(s_lst)):
            t1 = t1_lst[k1]
            s = s_lst[k2]
            k = k1 * len(s_lst) + k2
            Rs = np.where(surv > s+.05)[0]
            nk = len(Rs)
            time_new = np.array([[time[i][j][np.where(time[i][j] <= s)[0]]
                                  for j in range(nY)] for i in Rs])
            nn_new = np.array([[len(time_new[i][j])
                                for j in range(nY)] for i in range(nk)])
            Y_new = np.array([np.concatenate([Y[i][j][np.where(time[i][j] <= s)[0]]
                                              for j in range(nY)], axis=0)[:, None] for i in Rs])

            time_newh = np.array([[time[i][j][range(np.where(time[i][j] > s)[0][0], np.where(time[i][j] <= s+t1)[0][-1] + 1)]
                                  for j in range(nY)] for i in Rs])


            nn_newh = np.array([[len(time_newh[i][j])
                                for j in range(nY)] for i in range(nk)])
            Y_newh = np.array([np.concatenate([Y[i][j][range(np.where(time[i][j] > s)[0][0], np.where(time[i][j] <= s+t1)[0][-1] + 1)]
                                              for j in range(nY)], axis=0) for i in Rs])

            if q > 0:
                V_new = np.array(list(map(
                    lambda y: block_diag(*list(map(
                        lambda x: np.column_stack((PolyBasis(x, 1), TruncPolyBasis(x, knots, deg))),
                        y))),
                    time_new)))
            else:
                V_new = np.array(list(map(
                    lambda y: block_diag(*list(map(
                        lambda x: PolyBasis(x, 1),
                        y))),
                    time_new)))

            oneX = np.column_stack((np.ones(n), X))
            U_new = np.array([block_diag(*[np.column_stack((np.kron(np.ones((nn_new[i][j], 1), dtype=int),
                                                                    oneX[i]),  # intX = [1,X1,X2]
                                                            PolyBasis(time_new[i][j], deg, False)))
                                           for j in range(nY)])
                              for i in range(nk)])

            sigma_e2_inv = 1 / sigma_e ** 2

            Sigma_c = Sigma_b
            if q > 0:
                for j in range(nY):
                    Sigma_c = np.insert(Sigma_c, np.repeat(2 * (j + 1) + q * j, q), 0, axis=0)
                    Sigma_c = np.insert(Sigma_c, np.repeat(2 * (j + 1) + q * j, q), 0, axis=1)
                    idxj = np.arange(2 * (j + 1) + q * j, (2 + q) * (j + 1))
                    Sigma_c[idxj[:, None], idxj[None, :]] = np.diag(np.repeat(sigma_u[j] ** 2, q))
            Sigma_cinv = np.linalg.inv(Sigma_c)

            ScXY_new = np.array(list(map(
                lambda x, y: np.linalg.inv((
                        x.T.dot(np.diag(np.repeat(sigma_e2_inv, y))).dot(x) +
                        Sigma_cinv)), V_new, nn_new)))
            SinvDSY_new = np.array(list(map(
                lambda x, y, z, w: z.dot(x.T.dot(np.diag(np.repeat(sigma_e2_inv, y))).dot(w)),
                V_new, nn_new, ScXY_new, Y_new)))
            SinvDSU_new = np.array(list(map(lambda x, y, z, w:
                                        z.dot(x.T.dot(np.diag(np.repeat(sigma_e2_inv, y))).dot(w)),
                                        V_new, nn_new, ScXY_new, U_new)))

            mu_c_pos_new = np.array(SinvDSY_new.squeeze(2) - np.dot(SinvDSU_new, beta))

            Yhat_newh = [[[]] * nY for _ in range(nk)]

            for i in range(nk):
                for j in range(nY):
                    betaj = beta1[j]
                    if q > 0:
                        Dji_new = np.column_stack((PolyBasis(time_newh[i][j], min(deg, 1)),
                                                   TruncPolyBasis(time_newh[i][j], knots, deg)))
                    else:
                        Dji_new = PolyBasis(time_newh[i][j], min(deg, 1))
                    Uji_new = np.column_stack((np.kron(np.ones((nn_newh[i][j], 1), dtype=int),
                                                       np.insert(X[i], 0, 1)),  # intX = [1,X1,X2]
                                               PolyBasis(time_newh[i][j], deg, False)))

                    Yhat_newh[i][j] = np.dot(Uji_new, betaj) + np.dot(Dji_new, mu_c_pos_new[i][idx[j]]).T
                    # plt.plot(dat['Y'][i][j]);plt.plot(Yhat[i][j]);plt.show()

            N_new = np.sum(nn_newh)
            mse[k] = np.sum([np.sum((Y_newh[i] - np.concatenate(Yhat_newh[i])) ** 2) for i in range(nk)]) / N_new
            mae[k] = np.sum([np.sum(np.absolute(Y_newh[i] - np.concatenate(Yhat_newh[i]))) for i in range(nk)]) / N_new

    return mse, mae


def Accuracy(dat, stage2est):
    """Use time-dependent ROC for survival accuracy; MSE for longitudinal accuracy"""
    auc_surv_accuracy, bs_surv_accuracy, _, _ = surv_accuracy(dat, stage2est, plot_ROC = False)
    mse_long_accuracy, mae_long_accuracy = long_accuracy(dat, stage2est)
    accuracy = {'mse': mse_long_accuracy,
                'mae': mae_long_accuracy,
                'auc': auc_surv_accuracy,
                'bs': bs_surv_accuracy}
    return accuracy
