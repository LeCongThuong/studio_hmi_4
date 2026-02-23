\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\usepackage{algorithm}
\usepackage{algpseudocode}

\title{From Pixels to Rig-Consistent 3D Humans:\\
A Three-Component Multi-View Framework}
\author{Technical Report}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This paper presents a focused three-component framework for multi-view 3D human reconstruction.
The method progresses from per-view pixel understanding, to robust geometric fusion across cameras, and finally to rig-constrained optimization for anatomically coherent 3D outputs.
We emphasize the central modeling choices that govern quality: robust residual design, confidence-weighted supervision, similarity alignment, and separation of dynamic pose from static identity attributes.
\end{abstract}

\section{Introduction}
Multi-view human reconstruction can be viewed as a sequence of progressively stronger representations: pixels provide raw evidence, per-view inference provides semantic structure, triangulation produces cross-view 3D consensus, and rig optimization enforces physically plausible articulation.
This document keeps the presentation centered on those three components and their mathematical connection.

\section{Component I: Pixel-to-Perceptual Inference}
For each frame $t$ and camera $v$, an inference model maps the input image $I_t^{(v)}$ to structured outputs,
\begin{equation}
\mathbf{z}_t^{(v)} = f_{\theta}(I_t^{(v)}),
\end{equation}
where $\mathbf{z}_t^{(v)}$ contains 2D landmarks $\mathbf{u}_{t}^{(v)} \in \mathbb{R}^{K\times2}$, coarse 3D landmarks $\tilde{\mathbf{X}}_{t}^{(v)} \in \mathbb{R}^{K\times3}$, and latent rig parameters.
The key role of this stage is representational: it converts appearance into semantically aligned human structure that can later be fused geometrically.
These predictions are not final geometry; they are view-specific hypotheses carrying both signal and uncertainty.

\section{Component II: Multi-View Triangulation}
Given calibrated cameras and per-view observations $\mathbf{u}_{j,t}^{(v)}$ for landmark $j$, geometric supervision is obtained by robust reprojection minimization:
\begin{equation}
\mathbf{X}_{j,t}^{*}
=
\arg\min_{\mathbf{X}}
\sum_{v \in \mathcal{V}_{j,t}}
\rho\!\left(
\left\|
\mathbf{u}_{j,t}^{(v)} - \pi_v(\mathbf{X})
\right\|_2
\right),
\end{equation}
where $\pi_v(\cdot)$ is the calibrated projection model and $\rho(\cdot)$ is a robust penalty.
Robustness is essential because a small number of 2D outliers can otherwise dominate the solution.

In practice, each landmark is estimated through a formal four-step procedure:
\begin{enumerate}[leftmargin=1.2em]
    \item generate candidate 3D points by pairwise linear triangulation over view pairs,
    \item score candidates by cross-view robust reprojection residuals and select the best candidate,
    \item infer inlier views using thresholded residuals,
    \item run nonlinear refinement (damped least-squares) on the inlier set.
\end{enumerate}
The inlier set is defined as
\begin{equation}
\mathcal{I}_{j,t}
=
\left\{
 v \in \mathcal{V}_{j,t}
\;\middle|\;
\left\|
\mathbf{u}_{j,t}^{(v)} - \pi_v(\mathbf{X}_{j,t}^{\mathrm{init}})
\right\|_2 < \tau
\right\},
\end{equation}
and refinement solves
\begin{equation}
\mathbf{X}_{j,t}^{*}
=
\arg\min_{\mathbf{X}}
\sum_{v \in \mathcal{I}_{j,t}}
 w_{j,t}^{(v)}
\left\|
\mathbf{u}_{j,t}^{(v)} - \pi_v(\mathbf{X})
\right\|_2^2.
\end{equation}
The resulting refined set $\{\mathbf{X}_{j,t}^{*}\}$ provides pseudo-ground-truth supervision for rig fitting, while weights $w_{j,t}^{(v)}$ reduce the impact of weak observations.

\section{Component III: Rig-Constrained Optimization}
Let the rig state be decomposed as
\begin{equation}
\bm{\theta}_t = (\mathbf{p}_t, \mathbf{a}_t),
\end{equation}
with dynamic pose parameters $\mathbf{p}_t$ and non-pose attributes $\mathbf{a}_t$ (shape, scale, and related identity factors).
For a single subject, temporal consistency improves when non-pose attributes are fixed from a reference estimate,
\begin{equation}
\mathbf{a}_t \equiv \mathbf{a}_{\text{ref}},
\end{equation}
so articulation changes over time while identity remains stable.

Given a rig forward model $\mathcal{F}$, predicted supervision landmarks are
\begin{equation}
\hat{\mathbf{Y}}_{t,m} = \mathcal{F}_m(\mathbf{p}_t, \mathbf{a}_t), \quad m\in\mathcal{M}.
\end{equation}
Before residual evaluation, predictions are aligned to triangulated targets with a weighted similarity transform,
\begin{equation}
\bar{\mathbf{Y}}_{t,m} = s_t \mathbf{R}_t \hat{\mathbf{Y}}_{t,m} + \mathbf{t}_t,
\end{equation}
which separates global rigid mismatch from local articulation error and improves optimization conditioning.

The per-frame objective is
\begin{equation}
\mathcal{L}_t
=
\mathcal{L}_{\text{data},t}
+ \lambda_{\text{reg}}\mathcal{L}_{\text{reg},t}
+ \lambda_{\text{temp}}\mathcal{L}_{\text{temp},t}
+ \lambda_{\text{vel}}\mathcal{L}_{\text{vel},t},
\end{equation}
with data term
\begin{equation}
\mathcal{L}_{\text{data},t}
=
\frac{
\sum_{m\in\mathcal{M}} w_{t,m}\,
\rho_{\delta}\!\left(
\|\bar{\mathbf{Y}}_{t,m} - \mathbf{Y}_{t,m}\|_2
\right)}
{\sum_{m\in\mathcal{M}} w_{t,m} + \epsilon}.
\end{equation}
Regularization is defined by
\begin{align}
\mathcal{L}_{\text{reg},t} &= \|\mathbf{p}_t - \mathbf{p}_t^{0}\|_2^2, \\
\mathcal{L}_{\text{temp},t} &= \|\mathbf{p}_t - \mathbf{p}_{t-1}\|_2^2, \\
\mathcal{L}_{\text{vel},t} &= \|(\mathbf{p}_t-\mathbf{p}_{t-1})-(\mathbf{p}_{t-1}-\mathbf{p}_{t-2})\|_2^2.
\end{align}
Robust penalties limit outlier dominance, confidence weights modulate supervision reliability, and temporal terms suppress jitter while preserving motion continuity.

For constrained regions (e.g., lower body), hard masking can enforce fixed dimensions:
\begin{equation}
\mathbf{p}^{\text{eff}}_t
=
\mathbf{p}_t \odot \mathbf{m}
+
\mathbf{p}^{\text{fix}}_t \odot (1-\mathbf{m}),
\end{equation}
where $\mathbf{m}\in\{0,1\}^d$ selects optimized entries and freezes the rest exactly.

\section{End-to-End View}
\begin{algorithm}[h]
\caption{Three-Component Reconstruction}
\begin{algorithmic}[1]
\Require synchronized multi-view images and camera calibration
\For{each frame $t$}
    \State infer per-view perceptual outputs
    \State triangulate and refine supervised 3D landmarks
    \State optimize rig parameters against triangulated targets
\EndFor
\State return temporally consistent 3D keypoints and mesh-level outputs
\end{algorithmic}
\end{algorithm}

\section{Conclusion}
The framework is a direct progression from pixels to perception, from perception to geometry, and from geometry to rig-consistent optimization.
Triangulation provides robust 3D supervision, and the optimization objective converts that supervision into stable articulated motion with anatomical consistency.
The key practical principle is to let pose vary over time while keeping identity-defining non-pose attributes fixed for a single person.

\end{document}
