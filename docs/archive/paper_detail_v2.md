\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\usepackage{algorithm}
\usepackage{algpseudocode}

\title{Detailed Technical Report (V2):\\
Robust Multi-View Human Reconstruction with Specialized Hand Fusion}
\author{Technical Documentation}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This document describes the implemented V2 pipeline for multi-view human reconstruction.
The system combines a three-stage architecture with an optional external specialized hand model workflow:
(i) per-view SAM-3D inference and hand-keypoint fusion, (ii) robust triangulation with inlier-gated refinement, and (iii) rig-constrained optimization with quality gates, bad-frame handling, and sequence-level recovery/smoothing.
The report focuses on the current code behavior, interfaces, objective functions, and runtime controls.
\end{abstract}

\section{Pipeline Summary}
The current pipeline is orchestrated by a sequence runner and consists of:
\begin{enumerate}[leftmargin=1.2em]
    \item \textbf{Stage 0 (optional):} specialized hand precompute in a separate environment.
    \item \textbf{Stage 1:} per-view SAM-3D inference, optional hand fusion in 2D.
    \item \textbf{Stage 2:} robust multi-view triangulation on an MHR subset.
    \item \textbf{Stage 3:} MHR optimization to triangulated 3D targets.
\end{enumerate}

Sequence-level logic includes:
\begin{itemize}[leftmargin=1.2em]
    \item quality classification (\texttt{ok}, \texttt{bad\_loss}, failure statuses),
    \item optional retry for bad optimization (default retry count is zero),
    \item recovery for missing/bad frames by interpolation or bounded copy,
    \item optional smoothing and sequence summary export.
\end{itemize}

\section{Data Layout and Stage Contracts}
\subsection{Input Layout}
For frame index $t$ and camera $c$, expected image layout is:
\begin{equation}
\texttt{image\_folder}/t/c.\texttt{jpg}
\end{equation}
with at least two cameras per valid frame.

\subsection{Main Outputs}
Under \texttt{output\_root}:
\begin{itemize}[leftmargin=1.2em]
    \item \texttt{inference/npy/...}: per-view Stage-1 dictionaries.
    \item \texttt{triangulation/.../triangulated.npz}: Stage-2 robust 3D subset.
    \item \texttt{optimization/.../opt\_out.npy}: Stage-3 optimized output.
    \item \texttt{optimization/.../opt\_out\_smoothed.npy}: optional smoothed output.
    \item \texttt{sequence\_summary.json}: per-frame status and metrics.
\end{itemize}

\subsection{Inter-Stage Contract}
\begin{itemize}[leftmargin=1.2em]
    \item Stage-1 to Stage-2: \texttt{pred\_keypoints\_2d} with shape $(70,2)$ or $(N,70,2)$.
    \item Stage-2 to Stage-3: \texttt{points3d\_refined} $(M,3)$, \texttt{subset\_indices}, \texttt{inlier\_mask}, per-camera reprojection diagnostics.
    \item Stage-3 output: optimized SAM-style dictionary and \texttt{opt\_*} diagnostics.
\end{itemize}

\section{Stage 0: External Specialized Hand Precompute}
When specialized hand models are not available inside the SAM environment, the system supports external precompute.
\begin{itemize}[leftmargin=1.2em]
    \item Script: \texttt{run\_wilor\_precompute.py}.
    \item Per-image output: \texttt{<out>/<rel\_dir>/<image\_stem>.npy}.
    \item Each file stores \texttt{detections} with fields:
    \texttt{is\_right}, \texttt{hand\_bbox}, \texttt{pred\_keypoints\_2d} $(21,2)$.
\end{itemize}

This decouples environments:
\begin{itemize}[leftmargin=1.2em]
    \item WiLoR environment runs hand inference.
    \item SAM pipeline environment consumes saved detections.
\end{itemize}

\section{Stage 1: SAM Inference with Hand Fusion}
\subsection{Per-View Base Inference}
For each image $I_t^{(c)}$, SAM-3D predicts:
\begin{equation}
\mathbf{z}_t^{(c)} = f_{\theta}(I_t^{(c)}),
\end{equation}
including 2D keypoints, 3D keypoints, vertices, and rig-related parameters.

\subsection{Hand Fusion Inputs}
Let $\mathbf{u}^{\text{sam}}_{t,c,k}\in\mathbb{R}^2$ be SAM 2D keypoint $k\in\{0,\dots,69\}$.
Let $\mathbf{h}^{\text{sp}}_{t,c,j}\in\mathbb{R}^2$ be specialized hand keypoint $j\in\{0,\dots,20\}$.

Mapping functions convert specialized 21-joint order to MHR70 right/left indices:
\begin{equation}
m_{\text{R}}(j),\;m_{\text{L}}(j)\in\{0,\dots,69\}.
\end{equation}

\subsection{Candidate Selection per Side}
For each side (right/left), among candidate detections:
\begin{equation}
d = \left\|\mathbf{h}^{\text{sp}}_{\text{wrist}} - \mathbf{u}^{\text{sam}}_{\text{wrist}}\right\|_2.
\end{equation}
The selected candidate minimizes $d$ under wrist-distance gating:
\begin{equation}
d < \tau_{\text{wrist}}.
\end{equation}
If SAM wrist is invalid, a fallback heuristic prefers larger hand box area.

\subsection{Replacement Rule}
If a side candidate is accepted, mapped hand points replace SAM hand points:
\begin{equation}
\mathbf{u}_{t,c,m(j)} =
\begin{cases}
\mathbf{h}^{\text{sp}}_{t,c,j}, & \text{if finite and accepted},\\
\mathbf{u}^{\text{sam}}_{t,c,m(j)}, & \text{otherwise}.
\end{cases}
\end{equation}
Wrist replacement is optional (\texttt{replace\_wrist\_with\_specialized}).

The stage also writes source metadata:
\begin{itemize}[leftmargin=1.2em]
    \item \texttt{pred\_keypoints\_2d\_source} (length 70 mask),
    \item \texttt{specialized\_hand\_status\_right/left},
    \item \texttt{specialized\_hand\_num\_points}, \texttt{specialized\_hand\_used}.
\end{itemize}

\subsection{Fusion Debug Visualization}
When enabled, Stage-1 saves overlays that compare:
\begin{itemize}[leftmargin=1.2em]
    \item SAM hand points before fusion,
    \item replaced specialized points,
    \item fallback points retained from SAM.
\end{itemize}
This is intended for quick visual validation of per-frame fusion quality.

\section{Stage 2: Robust Multi-View Triangulation}
\subsection{Supervised Subset}
Triangulation uses a high-utility MHR subset:
\begin{itemize}[leftmargin=1.2em]
    \item right-hand keypoints (20),
    \item left-hand keypoints (20),
    \item shoulders, elbows, wrists (6),
\end{itemize}
so $M=46$ points per frame.

\subsection{Camera Model}
For camera $v$:
\begin{equation}
\mathbf{x}^{(v)}_{\text{cam}} = R_v\mathbf{X}+t_v,\quad
\mathbf{u}^{(v)}=\pi_v(\mathbf{X};K_v,D_v).
\end{equation}
DLT initialization uses undistorted normalized rays; residuals are evaluated in pixels through full projection.

\subsection{Step A: Pairwise Candidate Generation}
For each valid camera pair, generate $\mathbf{X}$ by linear triangulation (DLT).

\subsection{Step B: Robust Candidate Scoring}
For candidate $\mathbf{X}$, per-view reprojection residual magnitude is:
\begin{equation}
r_v(\mathbf{X})=\left\|\mathbf{u}^{(v)}-\pi_v(\mathbf{X})\right\|_2.
\end{equation}
Aggregate score options:
\begin{align}
S_{\text{median}}(\mathbf{X}) &= \mathrm{median}\{r_v\},\\
S_{\text{trimmed}}(\mathbf{X}) &= \frac{1}{K-1}\sum_{k=1}^{K-1} r_{(k)},\\
S_{\text{huber}}(\mathbf{X}) &= \sum_v \rho_\delta(r_v),
\end{align}
with Huber
\begin{equation}
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2, & r\le\delta,\\
\delta\left(r-\frac{1}{2}\delta\right), & r>\delta.
\end{cases}
\end{equation}
Select candidate with minimum robust score.
If pair candidates fail, fall back to all-view DLT.

\subsection{Step C: Inlier View Gating}
From selected initialization $\mathbf{X}^{\text{init}}$:
\begin{equation}
\mathcal{I}_{j,t}=\left\{v\;\middle|\;\left\|\mathbf{u}^{(v)}_{j,t}-\pi_v(\mathbf{X}^{\text{init}}_{j,t})\right\|_2<\tau\right\}.
\end{equation}
If fewer than two inliers are found but multiple views exist, force the two smallest-error views.
Optional reseed recomputes DLT from inliers.

\subsection{Step D: Inlier-Only LM Refinement}
For each point:
\begin{equation}
\mathbf{X}^{*}_{j,t}=
\arg\min_{\mathbf{X}}
\sum_{v\in\mathcal{I}_{j,t}}
\left\|\mathbf{u}^{(v)}_{j,t}-\pi_v(\mathbf{X})\right\|_2^2.
\end{equation}
Levenberg--Marquardt update:
\begin{equation}
\Delta\mathbf{X}=-(J^\top J+\lambda I)^{-1}J^\top\mathbf{r},\quad
\mathbf{X}\leftarrow \mathbf{X}+\Delta\mathbf{X}.
\end{equation}
Optional robust LM applies residual weighting with Huber-style weights.

\subsection{Triangulation Output}
\texttt{triangulated.npz} contains:
\begin{itemize}[leftmargin=1.2em]
    \item \texttt{points3d\_refined}, \texttt{subset\_indices}, \texttt{subset\_names},
    \item \texttt{inlier\_mask} $(M,V)$,
    \item per-camera reprojection errors (init and refined),
    \item robust-scoring and inlier-threshold metadata.
\end{itemize}

\section{Stage 3: Rig-Constrained Optimization}
\subsection{Variables and Forward Model}
State decomposition:
\begin{equation}
\bm{\theta}_t=(\mathbf{p}_t,\mathbf{h}_t,\mathbf{s}_t,\mathbf{b}_t,\mathbf{e}_t),
\end{equation}
where:
\begin{itemize}[leftmargin=1.2em]
    \item $\mathbf{p}_t$: body pose parameters,
    \item $\mathbf{h}_t$: hand108 parameters,
    \item $\mathbf{s}_t,\mathbf{b}_t,\mathbf{e}_t$: scale, shape, expression blocks.
\end{itemize}

Forward kinematics produces landmarks:
\begin{equation}
\hat{\mathbf{Y}}_{t,m}=\mathcal{F}_m(\bm{\theta}_t),\quad m\in\mathcal{M}.
\end{equation}

\subsection{View-Based Initialization}
Each view provides an initialization candidate.
Candidates are ranked by weighted aligned 3D residual to triangulated targets, with triangulation mean pixel error as tie-break.

\subsection{Fixed Non-Pose Parameters}
For single-person sequences, non-pose blocks can be fixed from a reference frame/camera:
\begin{equation}
\mathbf{s}_t\equiv\mathbf{s}_{\text{ref}},\quad
\mathbf{b}_t\equiv\mathbf{b}_{\text{ref}},\quad
\mathbf{e}_t\equiv\mathbf{e}_{\text{ref}}.
\end{equation}
Hand block can also be fixed if explicitly enabled.

\subsection{Masking and Lower-Body Freeze}
Binary mask $\mathbf{m}$ defines trainable dimensions:
\begin{equation}
\mathbf{p}^{\text{eff}}_t=
\mathbf{p}_t\odot\mathbf{m}+\mathbf{p}^{\text{fix}}_t\odot(1-\mathbf{m}).
\end{equation}
Hand/jaw-related dimensions are masked by default; optional lower-body freeze extends the frozen set.
Frozen dimensions are re-clamped every step.

\subsection{Aligned Data Loss}
Similarity transform aligns prediction to triangulated targets:
\begin{equation}
\bar{\mathbf{Y}}_{t,m}=s_tR_t\hat{\mathbf{Y}}_{t,m}+t_t.
\end{equation}
By default, alignment is estimated from stable anchors (shoulders, elbows, wrists, optional neck/acromion), while loss is computed on all valid supervised points.

Data term:
\begin{equation}
\mathcal{L}_{\text{data},t}=
\frac{
\sum_{m\in\mathcal{M}} w_{t,m}\,\rho_\delta\!\left(\left\|\bar{\mathbf{Y}}_{t,m}-\mathbf{Y}_{t,m}\right\|_2\right)
}{
\sum_{m\in\mathcal{M}} w_{t,m}+\epsilon
}.
\end{equation}
Weights $w_{t,m}$ are derived from triangulation inlier reliability (\texttt{inlier\_mask} averaged across views), then sanitized with finite-point checks.

\subsection{Total Objective}
\begin{equation}
\mathcal{L}_t=
\mathcal{L}_{\text{data},t}
\lambda_{\text{pose}}\mathcal{L}_{\text{pose},t}
\lambda_{\text{temp}}\mathcal{L}_{\text{temp},t}
\lambda_{\text{vel}}\mathcal{L}_{\text{vel},t}
\lambda_{\text{acc}}\mathcal{L}_{\text{acc},t}
\lambda_{\text{hand}}\mathcal{L}_{\text{hand},t},
\end{equation}
with:
\begin{align}
\mathcal{L}_{\text{pose},t} &= \|\mathbf{p}_t-\mathbf{p}^0_t\|_2^2,\\
\mathcal{L}_{\text{temp},t} &= \|\mathbf{p}_t-\mathbf{p}_{t-1}\|_2^2,\\
\mathcal{L}_{\text{vel},t} &= \|\mathbf{p}_t-\hat{\mathbf{p}}^{\text{vel}}_t\|_2^2,\\
\mathcal{L}_{\text{acc},t} &= \|(\mathbf{p}_t-\mathbf{p}_{t-1})-(\mathbf{p}_{t-1}-\mathbf{p}_{t-2})\|_2^2,\\
\mathcal{L}_{\text{hand},t} &= \|\mathbf{h}_t-\mathbf{h}^0_t\|_2^2.
\end{align}

In current defaults, $\lambda_{\text{vel}}=\lambda_{\text{acc}}=0$, so velocity and acceleration terms are inactive unless explicitly enabled.

\subsection{Optimization Procedure}
\begin{itemize}[leftmargin=1.2em]
    \item Optimizer: Adam on trainable pose (and optional hand108).
    \item Stabilization: gradient clipping and frozen-gradient zeroing.
    \item Stops: minimum-iteration gate, patience on non-improvement, and divergence ratio.
    \item Best iterate (lowest total loss) is restored before final export.
\end{itemize}

\subsection{Bad-Frame Classification}
Each frame stores:
\begin{itemize}[leftmargin=1.2em]
    \item \texttt{best\_loss}, \texttt{final\_loss},
    \item \texttt{best\_data\_loss}, \texttt{final\_data\_loss}.
\end{itemize}
A frame is marked bad if any condition holds:
\begin{equation}
\texttt{best\_loss}>\tau_{\text{loss}},
\quad
\texttt{best\_data\_loss}>\tau_{\text{data}},
\quad
\frac{\texttt{final\_loss}}{\max(\texttt{best\_loss},\epsilon)}>\gamma,
\quad
\frac{\texttt{final\_data\_loss}}{\max(\texttt{best\_data\_loss},\epsilon)}>\gamma.
\end{equation}

\section{Sequence Orchestration and Recovery}
\subsection{Frame Discovery and Stage Control}
The orchestrator discovers frame folders, enforces a minimum-view requirement, and can skip/reuse stage artifacts.

\subsection{Temporal Guard}
Temporal priors are disabled after a long non-good streak:
\begin{equation}
\texttt{stale\_run}\ge\texttt{max\_stale\_temporal\_frames}.
\end{equation}
This avoids stale-state lock-in.

\subsection{Bad-Frame Retry}
Retry logic exists but is optional. Current default is:
\begin{equation}
\texttt{bad\_frame\_max\_retries}=0,
\end{equation}
so retries are disabled unless explicitly set.

\subsection{Recovery Policy}
Bad/missing optimization outputs are treated as unusable and replaced by:
\begin{enumerate}[leftmargin=1.2em]
    \item interpolation when valid neighbors exist on both sides,
    \item bounded one-sided copy when only one side exists and distance is within \texttt{max\_edge\_recovery\_copy\_span}.
\end{enumerate}
Recovered frames are explicitly labeled as recovered and their copied optimization metrics are cleared to avoid misleading reports.

\subsection{Optional Sequence Smoothing}
Post-processing can apply nearest-valid fill, temporal median filtering, outlier suppression, and bidirectional EMA on key numeric fields.

\section{Important Runtime Controls}
\begin{center}
\begin{tabular}{>{\ttfamily}p{0.38\linewidth} p{0.55\linewidth}}
\toprule
Control & Effect \\
\midrule
enable\_specialized\_hand\_fusion & Enables hand keypoint replacement in Stage-1. \\
specialized\_hand\_source & \texttt{precomputed} or \texttt{live}. \\
specialized\_hand\_input\_root & Root of external hand detections for precomputed mode. \\
specialized\_hand\_wrist\_max\_dist\_px & Wrist-distance gate for accepting specialized candidates. \\
specialized\_hand\_debug\_vis & Saves Stage-1 SAM-vs-specialized fusion overlays. \\
score\_type, huber\_delta & Robust scoring mode for triangulation candidate selection. \\
inlier\_thresh & Reprojection threshold for inlier view gating. \\
robust\_lm, robust\_lm\_delta & Enables robust residual weighting in LM refinement. \\
fixed\_mhr\_param\_frame\_idx, fixed\_mhr\_param\_cam & Fixes non-pose blocks from one reference frame/camera. \\
fixed\_hand\_pose\_from\_reference & Also fixes hand pose from reference if enabled. \\
optimize\_hand\_pose & Enables hand108 optimization in Stage-3. \\
use\_anchor\_similarity & Estimates similarity from stable anchors instead of all supervised points. \\
freeze\_lower\_body & Freezes lower-body dimensions to reduce stationary jitter. \\
bad\_frame\_max\_retries & Optional bad-frame retries (default 0). \\
max\_edge\_recovery\_copy\_span & Bounds one-sided recovery copy distance. \\
\bottomrule
\end{tabular}
\end{center}

\section{Recommended V2 Configuration (Static Single-Person)}
\begin{enumerate}[leftmargin=1.2em]
    \item Use precomputed specialized hand detections and fuse in Stage-1.
    \item Fix non-pose identity blocks from a clean reference frame (front camera).
    \item Keep anchor-only similarity enabled.
    \item Enable lower-body freeze for stationary standing sequences.
    \item Keep velocity/acceleration weights at zero unless a specific motion scenario justifies them.
    \item Keep bad-frame retries disabled and rely on interpolation recovery.
\end{enumerate}

\section{Reproducibility Commands}
\subsection{External Hand Precompute}
\begin{verbatim}
python run_wilor_precompute.py \
  --image_folder /path/to/frames_root \
  --output_root /path/to/wilor_precomputed \
  --device cuda \
  --hand_conf 0.3 \
  --rescale_factor 2.5 \
  --debug_vis
\end{verbatim}

\subsection{End-to-End Pipeline}
\begin{verbatim}
python run_full_pipeline.py \
  --image_folder /path/to/frames_root \
  --output_root /path/to/pipeline_out \
  --cams left front right \
  --caliscope_toml /path/to/config.toml \
  --mhr_py mhr70.py \
  --checkpoint_path /path/to/model.ckpt \
  --mhr_path /path/to/mhr_model.pt \
  --hf_repo facebook/sam-3d-body-dinov3 \
  --enable_specialized_hand_fusion \
  --specialized_hand_source precomputed \
  --specialized_hand_model wilor \
  --specialized_hand_input_root /path/to/wilor_precomputed \
  --specialized_hand_debug_vis \
  --fixed_mhr_param_frame_idx 10 \
  --fixed_mhr_param_cam front \
  --freeze_lower_body
\end{verbatim}

\section{Limitations}
\begin{itemize}[leftmargin=1.2em]
    \item Accuracy depends on camera calibration and synchronization quality.
    \item Hand fusion quality depends on specialized detector robustness under occlusion and blur.
    \item Subset-based supervision may under-constrain body parts not represented in the triangulated subset.
    \item Recovery is a fallback for robustness, not a substitute for high-quality observations.
\end{itemize}

\section{Conclusion}
The implemented V2 pipeline provides a practical and controllable reconstruction workflow: specialized hand cues improve 2D evidence quality, robust triangulation improves 3D supervision reliability, and rig-constrained optimization enforces anatomical consistency with explicit diagnostics and recovery.
The main strength is sequence robustness with transparent frame-level status semantics.

\appendix
\section{Overleaf Usage}
This file is LaTeX source stored as \texttt{paper\_detail\_v2.md}.
For Overleaf:
\begin{enumerate}[leftmargin=1.2em]
    \item upload \texttt{paper\_detail\_v2.md},
    \item rename it to \texttt{main.tex},
    \item compile with pdfLaTeX.
\end{enumerate}

\end{document}
