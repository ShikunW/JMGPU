# JMGPU
Open source software for manuscript "GPU Accelerated Statistical Methods through a Deep Learning Framework"

Authors: Shikun Wang, Zhao Li, Lan Lan, Jieyi Zhao, Wenjin Zheng, Liang Li

### Description of the code
This code shows a GPU implementation of a joint model for multiple nonlinear longitudinal processes and time-to-event outcomes through Pytorch.

### Abstract
In longitudinal cohort studies, it is often of interest to predict the risk of a terminal clinical event using accumulating longitudinal predictor data among those patients who are still at-risk for the terminal event. The at-risk patient population may change over time, and so is the association between predictors and the outcome. This dynamic prediction problem has received increasing interest in the literature, but there remain computational challenges in its analysis. The widely used joint model of longitudinal and survival data often suffers intensive computation or excessive model fitting time, due to numerical optimization and the analytically intractable high-dimensional integral in the likelihood function. This problem is exacerbated when the model is fit to a large dataset or the model involves multiple longitudinal predictors with nonlinear trajectories. In this paper, we address this problem from an algorithmic perspective, by proposing a novel two-stage estimation procedure, and from a computing perspective, by using Graphics Processing Unit (GPU) programming. The latter is implemented through \texttt{PyTorch}, an emerging deep learning framework. Our numerical studies demonstrate that our proposed algorithm and software can substantially speed up the estimation of the joint model, particularly with large datasets. We also found that accounting for the nonlinearity in longitudinal predictor trajectories can improve the prediction accuracy in comparison to joint models that ignore nonlinearity. 
