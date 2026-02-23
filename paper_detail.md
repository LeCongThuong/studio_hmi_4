\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{array}
\usepackage{siunitx}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\usepackage{algorithm}
\usepackage{algpseudocode}

\title{Detailed Technical Report:\\
Three-Component Multi-View Human Reconstruction Pipeline}
\author{Technical Documentation}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This document provides a detailed description of a production-style multi-view human reconstruction pipeline composed of three components: (i) per-view pixel-to-perceptual inference, (ii) robust multi-view triangulation for geometric supervision, and (iii) rig-constrained optimization for final 3D motion and mesh outputs.
The report focuses on implementation-grounded details: stage interfaces, robust estimation procedures, loss construction, temporal priors, frame-level quality logic, recovery/smoothing behavior, and key controls that determine long-sequence stability.
Special attention is given to identity-consistent optimization via fixed non-pose rig attributes and to lower-body stabilization via hard parameter freezing.
\end{abstract}

\section{Introduction}
Multi-view human reconstruction in long sequences is difficult because each stage is affected by different failure sources: visual ambiguity at the perception level, geometric inconsistency across cameras, and optimizer drift in latent rig space.
A robust system therefore benefits from a staged decomposition where each component has a strict role and explicit intermediate outputs.

The pipeline described here follows a three-component design:
\begin{enumerate}[leftmargin=1.2em]
    \item \textbf{Component I: Pixel-to-Perceptual Inference} maps each image to human-structured predictions.
    \item \textbf{Component II: Multi-View Triangulation} fuses per-view 2D evidence into robust 3D supervision.
    \item \textbf{Component III: Rig-Constrained Optimization} fits a parametric rig to triangulated supervision with robust and temporal objectives.
\end{enumerate}
A sequence orchestrator executes these components frame by frame with quality classification, retries, bounded recovery, and optional smoothing.

\section{Problem Setup and Notation}
At time index $t$, let the set of available camera views be $\mathcal{V}_t$.
For each view $v \in \mathcal{V}_t$, the observed image is $I_t^{(v)}$.
Camera calibration provides intrinsic and extrinsic parameters used by projection operator $\pi_v(\cdot)$.

The pipeline outputs, per valid frame, include:
\begin{itemize}[leftmargin=1.2em]
    \item triangulated 3D supervision landmarks $\{\mathbf{X}_{j,t}^{*}\}$,
    \item optimized rig parameters $\bm{\theta}_t$,
    \item aligned 3D keypoints and mesh-level geometry,
    \item per-frame quality metrics and status labels.
\end{itemize}

\section{System Architecture and Data Flow}
\subsection{Frame-Level Data Contracts}
The main stage contracts are:
\begin{itemize}[leftmargin=1.2em]
    \item \textbf{I $\rightarrow$ II}: per-view 2D landmarks (typically $70\times2$ in pixel coordinates after normalization handling).
    \item \textbf{II $\rightarrow$ III}: refined subset 3D points, subset indices/names, per-view reprojection diagnostics, inlier mask.
    \item \textbf{III output}: optimized rig parameters and aligned geometry, plus optimization diagnostics.
\end{itemize}

\subsection{Subset Supervision Design}
Triangulation and optimization do not supervise all keypoints uniformly.
A high-utility subset is used:
\begin{itemize}[leftmargin=1.2em]
    \item right-hand landmarks (20),
    \item left-hand landmarks (20),
    \item shoulders, elbows, wrists (6).
\end{itemize}
Hence $M=46$ supervised landmarks per frame.

\section{Component I: Pixel-to-Perceptual Inference}
For each camera view, inference computes:
\begin{equation}
\mathbf{z}_t^{(v)} = f_{\theta}(I_t^{(v)}),
\end{equation}
where $\mathbf{z}_t^{(v)}$ includes 2D keypoints, coarse 3D keypoints, and latent rig parameter estimates.

\subsection{Single-Subject Selection}
When multiple candidates are present, one candidate is selected by policy (first, largest bbox, or fixed index).
This enforces a single-subject stream for downstream geometric and optimization consistency.

\subsection{Saved Artifacts}
Per-view artifacts store dictionary-style predictions used downstream.
Key fields include 2D/3D keypoints and rig-related parameters (body pose, hand pose, scale, shape, expression).
An additional metadata record preserves stage configuration identity for cache validation across runs.

\section{Component II: Robust Multi-View Triangulation}
\subsection{Camera Model and Residual Space}
For each camera $v$, calibration defines
\begin{equation}
\mathbf{x}^{(v)}_{\mathrm{cam}} = R_v\mathbf{X} + t_v,
\qquad
\mathbf{u}^{(v)} = \pi_v(\mathbf{X};K_v,D_v),
\end{equation}
with intrinsics $K_v$, distortion $D_v$, and extrinsics $(R_v,t_v)$.
The triangulation stage uses undistorted normalized points for linear initialization, but evaluates residuals in pixel space through full projection.

\subsection{Robust Landmark Objective}
For each subset landmark $j$ at frame $t$:
\begin{equation}
\mathbf{X}^{*}_{j,t}
=
\arg\min_{\mathbf{X}}
\sum_{v\in\mathcal{V}_{j,t}}
\rho\!\left(\left\|\mathbf{u}^{(v)}_{j,t} - \pi_v(\mathbf{X})\right\|_2\right),
\end{equation}
where $\mathcal{V}_{j,t}$ is the set of views with valid observations.

\subsection{Step A: Pairwise DLT Candidate Generation}
For each valid camera pair $(a,b)$, a homogeneous DLT candidate is formed:
\begin{equation}
A\mathbf{X}_h=0,
\qquad
A=
\begin{bmatrix}
x_aP_a^{(3,:)}-P_a^{(1,:)}\\
y_aP_a^{(3,:)}-P_a^{(2,:)}\\
x_bP_b^{(3,:)}-P_b^{(1,:)}\\
y_bP_b^{(3,:)}-P_b^{(2,:)}
\end{bmatrix},
\end{equation}
where $(x_v,y_v)$ are undistorted normalized coordinates and $P_v=[R_v\mid t_v]$.
The 3D point is recovered as $\mathbf{X}=\mathbf{X}_h^{1:3}/\mathbf{X}_h^4$.

\subsection{Step B: Robust Candidate Scoring}
For one candidate 3D point $\mathbf{X}$, we first compute a per-view reprojection error:
\begin{equation}
\hat{\mathbf{u}}_{j,t}^{(v)}=\pi_v(\mathbf{X}),\qquad
\mathbf{e}_{j,t}^{(v)}=\mathbf{u}_{j,t}^{(v)}-\hat{\mathbf{u}}_{j,t}^{(v)},\qquad
r_v=\left\|\mathbf{e}_{j,t}^{(v)}\right\|_2.
\end{equation}
Here $r_v$ is a scalar in pixels: how far the projected point is from the observed 2D keypoint in camera $v$.

We then reduce the set of errors $\{r_v\}$ to one scalar score $S(\mathbf{X})$.
The candidate with the smallest score is selected.

The pipeline supports three score choices:
\begin{align}
S_{\text{median}}(\mathbf{X}) &= \mathrm{median}(\{r_v\}),\\
S_{\text{trimmed}}(\mathbf{X}) &= \frac{1}{K-1}\sum_{k=1}^{K-1} r_{(k)},\\
S_{\text{huber}}(\mathbf{X}) &= \sum_v \rho_\delta(r_v),
\end{align}
where $r_{(k)}$ are residuals sorted from small to large and $K$ is the number of valid views for that landmark.
Intuitively:
\begin{itemize}[leftmargin=1.2em]
    \item $S_{\text{median}}$: robust to one or a few outlier views.
    \item $S_{\text{trimmed}}$: explicitly drops the single worst view error.
    \item $S_{\text{huber}}$: keeps all views but down-weights large residuals smoothly.
\end{itemize}

The Huber penalty is
\begin{equation}
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2,& r\le\delta,\\
\delta\left(r-\frac{1}{2}\delta\right),& r>\delta.
\end{cases}
\end{equation}
The minimum-score rule is applied only over valid pairwise candidates.
If no pairwise candidate is valid (candidate set is empty), the solver falls back to a single all-view DLT estimate.

\subsection{Step C: Inlier Set Construction}
Goal: keep only camera views that are geometrically consistent with the selected candidate $\mathbf{X}^{\mathrm{init}}_{j,t}$.

For each valid view $v$, compute its pixel reprojection error using the selected candidate:
\begin{equation}
r_v^{\mathrm{init}}=\left\|\mathbf{u}^{(v)}_{j,t}-\pi_v\left(\mathbf{X}^{\mathrm{init}}_{j,t}\right)\right\|_2.
\end{equation}
The inlier set is then defined as
\begin{equation}
\mathcal{I}_{j,t}
=
\left\{v\in\mathcal{V}_{j,t}
\;\middle|\;
r_v^{\mathrm{init}} < \tau
\right\}.
\end{equation}
Interpretation of $\\tau$: a view is trusted only if its reprojection error is below this pixel threshold.

To avoid under-constrained optimization, the pipeline enforces:
\\begin{enumerate}[leftmargin=1.2em]
    \\item if $|\\mathcal{I}_{j,t}|<2$ but at least two valid views exist, force the two smallest-error views as inliers;
    \\item optionally recompute a DLT seed using only inlier views before LM refinement.
\\end{enumerate}

\subsection{Step D: Inlier-Only LM Refinement}
Goal: update $\mathbf{X}^{\mathrm{init}}_{j,t}$ to a locally optimal $\mathbf{X}^{*}_{j,t}$ using only the inlier views from Step C.

The refinement objective is
\begin{equation}
\mathbf{X}^{*}_{j,t}
=
\arg\min_{\mathbf{X}}
\sum_{v\in\mathcal{I}_{j,t}}
\left\|\mathbf{u}^{(v)}_{j,t}-\pi_v(\mathbf{X})\right\|_2^2.
\end{equation}
At each iteration $q$, the solver stacks 2D residuals from inlier views into $\mathbf{r}_q$ and computes Jacobian $J_q$ (numerically in this implementation).
The Levenberg--Marquardt update is
\begin{equation}
\Delta\mathbf{X}_q = -(J_q^\top J_q + \lambda_q I)^{-1}J_q^\top\mathbf{r}_q,
\qquad
\mathbf{X}_{q+1}=\mathbf{X}_q+\Delta\mathbf{X}_q.
\end{equation}
The damping term $\lambda_q$ stabilizes updates when the local system is ill-conditioned.

With robust LM enabled, each inlier-view residual block is scaled by $\sqrt{w_v}$:
\begin{equation}
w_v=
\begin{cases}
1,& r_v\le\delta_{\mathrm{LM}},\\
\delta_{\mathrm{LM}}/r_v,& r_v>\delta_{\mathrm{LM}}.
\end{cases}
\end{equation}
Hence large residuals contribute less to the update, which reduces outlier influence during refinement.

\subsection{Outputs to Optimization}
Triangulation exports: refined 3D subset points, subset indices and names, per-point inlier mask, per-camera reprojections and errors (init/refined), and per-camera mean refined reprojection errors.

\section{Component III: Rig-Constrained Optimization}
\subsection{Variables and Forward Model}
The per-frame rig state is decomposed as
\begin{equation}
\bm{\theta}_t=(\mathbf{p}_t,\mathbf{a}_t),
\end{equation}
where $\mathbf{p}_t$ is the optimized pose vector (typically 133D), and $\mathbf{a}_t$ are non-pose attributes (hand, scale, shape, expression blocks).
Forward kinematics predicts supervised landmarks:
\begin{equation}
\hat{\mathbf{Y}}_{t,m}=\mathcal{F}_m(\mathbf{p}_t,\mathbf{a}_t),
\qquad m\in\mathcal{M}.
\end{equation}

\subsection{View Ranking and Initialization}
Each camera output provides one initialization candidate.
For each candidate, weighted Umeyama alignment to triangulated targets is computed:
\begin{equation}
\mathbf{Y} \approx s(\mathbf{X}R^\top)+t.
\end{equation}
The camera with lowest weighted aligned 3D residual is selected; triangulation mean pixel error is used as a tie-breaker.
This avoids unstable starts from poor view hypotheses.

\subsection{Identity-Consistent Non-Pose Parameters}
For single-person sequences, non-pose parameters are fixed from one reference frame/camera:
\begin{equation}
\mathbf{a}_t \equiv \mathbf{a}_{\mathrm{ref}}.
\end{equation}
This removes framewise body-scale and identity drift caused by unconstrained re-estimation of non-pose attributes.

\subsection{Masking and Hard Freezing}
Optimization uses a binary mask $\mathbf{m}\in\{0,1\}^d$:
\begin{equation}
\mathbf{p}^{\mathrm{eff}}_t
=
\mathbf{p}_t\odot\mathbf{m}
+
\mathbf{p}^{\mathrm{fix}}_t\odot(1-\mathbf{m}).
\end{equation}
The base mask freezes hand and jaw-related dimensions.
Optional lower-body freeze extends the frozen set with predefined lower-body indices.
Per iteration: frozen gradients are zeroed, then post-step clamping restores exact frozen values.

\subsection{Valid-Point and Weight Sanitization}
Point confidence comes from triangulation inlier masks.
Before each loss evaluation, the system:
\begin{itemize}[leftmargin=1.2em]
    \item removes non-finite target or prediction points,
    \item clips invalid weights to $[0,1]$,
    \item optionally falls back to uniform finite-point weighting if weighted valid points are too few.
\end{itemize}
This prevents degenerate optimization on sparse or corrupted frames.

\subsection{Detailed Loss Function}
For one frame $t$, optimization updates pose parameters by minimizing
\begin{equation}
\mathcal{L}_t
=
\mathcal{L}_{\mathrm{data},t}
+\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg},t}
+\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp},t}
+\lambda_{\mathrm{vel}}\mathcal{L}_{\mathrm{vel},t}
+\lambda_{\mathrm{acc}}\mathcal{L}_{\mathrm{acc},t}.
\end{equation}

Notation used below:
\begin{itemize}[leftmargin=1.2em]
    \item $m\in\mathcal{M}$: supervised landmark index in the triangulated subset.
    \item $\hat{\mathbf{Y}}_{t,m}$: predicted 3D landmark from forward kinematics (before alignment).
    \item $\mathbf{Y}_{t,m}$: triangulated target 3D landmark.
    \item $\bar{\mathbf{Y}}_{t,m}$: aligned prediction used for residual computation.
    \item $w_{t,m}\in[0,1]$: confidence weight (derived from inlier reliability).
    \item $\mathbf{p}_t$: current pose variable being optimized.
    \item $\mathbf{p}_t^0$: reference pose (initialization anchor for this frame).
    \item $\mathbf{p}_{t-1},\mathbf{p}_{t-2}$: previous accepted frame poses.
    \item $\hat{\mathbf{p}}^{\mathrm{vel}}_t$: velocity-based temporal target.
    \item $\delta$: Huber transition threshold, $\epsilon$: numerical stabilizer.
\end{itemize}

Residuals are computed in aligned space to remove global rigid mismatch:
\begin{equation}
\bar{\mathbf{Y}}_{t,m}=s_tR_t\hat{\mathbf{Y}}_{t,m}+t_t.
\end{equation}

Data term (robust weighted point matching):
\begin{equation}
\mathcal{L}_{\mathrm{data},t}
=
\frac{
\sum_{m\in\mathcal{M}}w_{t,m}\,\rho_\delta\!\left(\|\bar{\mathbf{Y}}_{t,m}-\mathbf{Y}_{t,m}\|_2\right)
}{\sum_{m\in\mathcal{M}}w_{t,m}+\epsilon},
\end{equation}
with Huber penalty
\begin{equation}
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2,& r\le\delta,\\
\delta\left(r-\frac{1}{2}\delta\right),& r>\delta.
\end{cases}
\end{equation}
Interpretation: small residuals are optimized quadratically; large residuals are linearized so outliers do not dominate.

Pose prior and temporal terms:
\begin{align}
\mathcal{L}_{\mathrm{reg},t} &= \|\mathbf{p}_t-\mathbf{p}_t^0\|_2^2,\\
\mathcal{L}_{\mathrm{temp},t} &= \|\mathbf{p}_t-\mathbf{p}_{t-1}\|_2^2,\\
\mathcal{L}_{\mathrm{vel},t} &= \|\mathbf{p}_t-\hat{\mathbf{p}}^{\mathrm{vel}}_t\|_2^2,\\
\mathcal{L}_{\mathrm{acc},t} &= \|(\mathbf{p}_t-\mathbf{p}_{t-1})-(\mathbf{p}_{t-1}-\mathbf{p}_{t-2})\|_2^2.
\end{align}
Intuition:
\begin{itemize}[leftmargin=1.2em]
    \item $\mathcal{L}_{\mathrm{reg}}$: keeps the solution close to a plausible initialization.
    \item $\mathcal{L}_{\mathrm{temp}}$: discourages frame-to-frame jumps.
    \item $\mathcal{L}_{\mathrm{vel}}$: encourages current pose to follow estimated motion direction.
    \item $\mathcal{L}_{\mathrm{acc}}$: suppresses sudden velocity change (jitter).
\end{itemize}

\subsection{Temporal Initialization and Velocity Target}
If previous good frames exist, initialization is not purely per-view; it blends view cue and temporal prediction:
\begin{equation}
\mathbf{p}_t^{\mathrm{init}}
=(1-\beta\mathbf{m})\odot\mathbf{p}^{\mathrm{view}}
+(\beta\mathbf{m})\odot\hat{\mathbf{p}}^{\mathrm{vel}}_t,
\qquad \beta\in[0,1].
\end{equation}
Here:
\begin{itemize}[leftmargin=1.2em]
    \item $\mathbf{p}^{\mathrm{view}}$: pose initialization from the selected best camera view.
    \item $\mathbf{m}$: optimization mask (frozen dimensions remain unaffected).
    \item $\beta$: blend strength; larger $\beta$ means stronger temporal pull.
\end{itemize}

Velocity target is
\begin{equation}
\hat{\mathbf{p}}^{\mathrm{vel}}_t
=
\mathbf{p}_{t-1}
+\eta(\mathbf{p}_{t-1}-\mathbf{p}_{t-2}),
\end{equation}
where $\eta$ is temporal extrapolation gain.
If history is unavailable, the optimizer falls back to view-only initialization.

\subsection{Numerical Optimization Loop}
Per iteration, the solver performs:
\begin{enumerate}[leftmargin=1.2em]
    \item build effective pose using freeze mask and fixed dimensions,
    \item run forward kinematics to get predicted landmarks,
    \item keep only valid supervised points and sanitized weights,
    \item solve similarity alignment $(s_t,R_t,t_t)$ for current prediction,
    \item evaluate total loss $\mathcal{L}_t$ and backpropagate,
    \item zero gradients on frozen dimensions, clip gradients, apply Adam step,
    \item re-clamp frozen dimensions exactly.
\end{enumerate}

Optimization stops when one of the following is triggered:
\begin{itemize}[leftmargin=1.2em]
    \item minimum-iteration condition is satisfied and no improvement persists for a patience window,
    \item loss diverges above a configured ratio relative to the best loss so far,
    \item maximum iteration budget is reached.
\end{itemize}
The best-loss iterate found during the loop is restored before final output generation.

\subsection{Final Similarity and Lower-Body Stability}
After optimization, final geometry is aligned by similarity transform $(s,R,t)$.
If lower-body freeze is enabled and previous-frame similarity is valid, reusing previous similarity is allowed:
\begin{equation}
(s_t,R_t,t_t) \leftarrow (s_{t-1},R_{t-1},t_{t-1}),
\end{equation}
which suppresses rigid-alignment jitter on static lower-body segments.

\subsection{Bad-Frame Classification}
After optimization, the frame is evaluated with four summary numbers:
\begin{itemize}[leftmargin=1.2em]
    \item \texttt{best\_loss}: smallest total objective value reached during iterations.
    \item \texttt{final\_loss}: total objective value at the last iteration/output.
    \item \texttt{best\_data\_loss}: smallest data-term value reached during iterations.
    \item \texttt{final\_data\_loss}: data-term value at the last iteration/output.
\end{itemize}

The frame is marked as bad if any of the following conditions is true:
\begin{equation}
\texttt{best\_loss}>\tau_{\mathrm{loss}},
\quad
\texttt{best\_data\_loss}>\tau_{\mathrm{data}},
\quad
\frac{\texttt{final\_loss}}{\max(\texttt{best\_loss},\epsilon)}>\gamma,
\end{equation}
and the same growth-ratio check is also applied to the data term:
\begin{equation}
\frac{\texttt{final\_data\_loss}}{\max(\texttt{best\_data\_loss},\epsilon)}>\gamma.
\end{equation}
Intuition: reject frames that never achieved a good fit (high best losses) or that became unstable by the end (large final-to-best growth).

\section{Sequence Orchestration and Reliability Logic}
\subsection{Frame Discovery and Stage Control}
The orchestrator discovers frame folders, enforces a minimum-view criterion, and runs enabled components.
Each stage can be independently skipped provided its required artifacts already exist.

\subsection{Temporal Priors Guard}
Temporal initialization/priors are disabled after a configurable run length of consecutive non-good frames.
This prevents stale temporal states from dominating optimization during prolonged degraded segments.

\subsection{Retry Policy for Bad Frames}
Bad-loss frames may be retried with:
\begin{itemize}[leftmargin=1.2em]
    \item lower learning rate,
    \item stronger temporal/velocity/acceleration weights,
    \item stronger temporal-initialization blend.
\end{itemize}
A retry is accepted when it becomes non-bad or improves best total/data loss.

\subsection{Recovery for Missing/Failed Frames}
If optimization output is absent or unusable, recovery uses:
\begin{enumerate}[leftmargin=1.2em]
    \item interpolation between nearest valid neighbors when both sides exist,
    \item one-sided copy from nearest valid side if within bounded span.
\end{enumerate}
Recovered frames are explicitly labeled as recovered and are prevented from carrying misleading copied optimization metrics.

\subsection{Sequence Smoothing}
Optional smoothing operates on selected numeric fields (pose, 3D keypoints, joint coords):
\begin{enumerate}[leftmargin=1.2em]
    \item nearest-valid fill,
    \item temporal median filtering,
    \item MAD-based outlier suppression,
    \item bidirectional EMA.
\end{enumerate}
This improves visual stability while preserving major motion trends.

\section{High-Level Algorithm}
\begin{algorithm}[h]
\caption{Detailed Sequence Reconstruction Pipeline}
\begin{algorithmic}[1]
\Require synchronized multi-view images, camera calibration, model checkpoints/config
\State run per-view inference to produce structured outputs
\For{each frame $t$ in sequence order}
    \State select available cameras; skip frame if below minimum view count
    \State triangulate supervised subset via robust pairwise init + inlier-gated LM refinement
    \State optimize rig pose using robust weighted aligned loss and temporal priors
    \If{frame marked bad and retries enabled}
        \State retry with stronger temporal constraints and conservative update schedule
    \EndIf
    \State update temporal history from good frames only
\EndFor
\State optionally recover failed/missing frames by interpolation or bounded one-sided copy
\State optionally smooth sequence outputs and export diagnostics/summary
\end{algorithmic}
\end{algorithm}

\section{Important Runtime Controls}
\begin{center}
\begin{tabular}{>{\ttfamily}p{0.35\linewidth} p{0.58\linewidth}}
\toprule
Control & Effect on behavior \\
\midrule
with\_scale & Enables scale in similarity alignment; useful when absolute scale mismatch exists. \\
score\_type, huber\_delta & Controls robust scoring for triangulation candidate selection. \\
inlier\_thresh & Sets inlier cutoff for triangulation view gating. \\
robust\_lm, robust\_lm\_delta & Enables robust residual weighting inside LM refinement. \\
w\_temporal, w\_temporal\_velocity, w\_temporal\_accel & Temporal smoothness/continuity pressure in pose optimization. \\
fixed\_mhr\_param\_frame\_idx, fixed\_mhr\_param\_cam & Freezes non-pose attributes from a reference view/frame for identity consistency. \\
freeze\_lower\_body & Hard-freezes lower-body pose dimensions to reduce stationary jitter. \\
max\_stale\_temporal\_frames & Disables temporal priors after long non-good streaks. \\
max\_edge\_recovery\_copy\_span & Bounds one-sided recovery copy distance to avoid long flat tails. \\
\bottomrule
\end{tabular}
\end{center}

\section{Failure Modes and Diagnostic Signals}
Common failure modes include:
\begin{itemize}[leftmargin=1.2em]
    \item sparse/invalid 2D observations causing unstable triangulation,
    \item inlier-weight collapse yielding under-constrained optimization,
    \item temporal lock-in under long runs of poor frames,
    \item body-identity drift if non-pose parameters vary framewise,
    \item lower-body world jitter if rigid similarity changes frame to frame despite static stance.
\end{itemize}

Useful diagnostics:
\begin{itemize}[leftmargin=1.2em]
    \item per-camera reprojection error before/after refinement,
    \item per-frame best/final total and data losses,
    \item bad-loss flags and recovery provenance,
    \item temporal reuse indicators and smoothing metadata.
\end{itemize}

\section{Practical Recommendations}
For single-person video pipelines:
\begin{enumerate}[leftmargin=1.2em]
    \item lock non-pose attributes from a clean reference frame/camera,
    \item enable lower-body freeze for static-stance segments,
    \item keep robust triangulation and robust optimization penalties enabled,
    \item tune temporal weights conservatively and keep stale-prior guard active,
    \item bound one-sided recovery to avoid motion flattening over long gaps.
\end{enumerate}

\section{Conclusion}
The pipeline achieves robust multi-view reconstruction by combining perceptual structure extraction, robust geometric consensus, and rig-aware optimization under temporal constraints.
Its strongest practical property is not only per-frame accuracy, but controlled long-sequence behavior through explicit quality gates, bounded temporal influence, identity-consistent parameterization, and transparent frame-level status accounting.

\end{document}
