# SPIRAL-PO code package

MATLAB / GNU Octave code that reproduces every number, table and figure of the
paper

> M. A. Ayoubi and S. Piramoon, *SPIRAL-PO: Symbolic Identification of Partially
> Observed Nonlinear Dynamics with Application to Rotating Machinery*,
> submitted to Nonlinear Dynamics (2026). Preprint: arXiv:XXXX.XXXXX.

SPIRAL-PO identifies, from the measured coordinates only, which mechanisms of
an unmeasured (hidden) subsystem leave a detectable signature in the measured
equations of motion. On a 4-DOF vertical rotor, only the lateral displacements
x and y are measured, and the disk tilt (alpha, beta) is hidden. The method
recovers the gyroscopic and cubic-bearing-stiffness signatures of the tilt
dynamics. In knock-out controls it does not admit a mechanism that has been
removed from the data.

No toolboxes are needed. The code was tested in GNU Octave 8.4 and is written
for MATLAB R2016b or later.

---

## 1. Quick start

```matlab
cd SPIRAL_PO_code
spiral_po_run('quick')     % 2-minute check: simulate and identify the nominal rotor
spiral_po_run              % everything (several hours); results go to results/
```

The quick check should print:

```
x-equation: selected Omega*ydot, (x^2+y^2)*x  (|t| > 2.73, 158 test functions)
y-equation: selected Omega*xdot, (x^2+y^2)*y  (|t| > 2.73, 158 test functions)
```

To use the functions interactively, add the folders to the path first with
`spiral_po_run('paths')`.

---

## 2. Folder layout

```
SPIRAL_PO_code/
  spiral_po_run.m          entry point (run order, paths, quick check)
  README.md                this file
  src/                     the method
    spiral_config.m        benchmark definition (Table 2): parameters, cases, seeds
    rotor_rhs.m            4-DOF rotor model (+ anisotropy, gyro knock-out, Bouc-Wen)
    simulate_rotor.m       synthetic record: RK4, noise on x and y, SG derivatives; re-noising
    sg_derivative.m        Savitzky-Golay derivatives with cross-validated window
    estimate_speed.m       spin speed from a displacement record (order tracking)
    build_library.m        physics seed and mechanism-labeled candidate library (Table 1)
    weak_form.m            the identification step: weak-form regression + Bonferroni test
    regression_baselines.m pointwise OLS variants and the moving-block bootstrap
    diagnostics.m          held-out gain, k-step prediction, conditioning,
                           residual predictability (Step 7), library capacity
    spiral_po_loop.m       gated network formulation (network, projection, Gates 1-2;
                           Sect. 2.2.9), used for the ablation study
    nn_fit.m               small residual network (Adam, early stopping)
    havok_baseline.m       delay-coordinate (HAVOK) baseline
  studies/                 one driver per section of the paper
    study_synthetic.m      Sect. 4.6-4.11: Tables 3-7, conditioning, constant speed
    study_ekf.m            Sect. 4.12-4.13: Fig. 4, Tables 9-11
    study_rig.m            Sect. 5: Tables 12-13, Fig. 6, speed check, controls, data check
  data/
    Experimental_Rotor_Data_SPIRAL_PO.csv   simultaneous x-y coast-down (two
                           laser sensors, Quanser DAQ, 1 kHz); its speed column
                           is an assumed constant deceleration, not a measurement
    SIG_9_9_2022.mat       single-sensor coast-downs (ILD1420, 4 kHz, one
                           lateral direction per run; separate session)
  results/                 all output (created on first run; results/cache holds
                           the simulated records)
```

---

## 3. Run order and what each step produces

Run from the package folder. Steps 1 to 5 must run in that order, because parts
`mc`, `coef` and `noise` reuse the records that parts `A` and `B` cache in
`results/cache`. The EKF and rig studies are independent of the rest.
`spiral_po_run` follows this order.

| # | Command | Produces | Paper | Time (Octave) |
|---|---|---|---|---|
| 1 | `study_synthetic('A')` | `step1_*` (nominal record, bootstrap tables, conditioning), `step2_ablations.csv`, `step3_knockouts.csv`, `summary_A.txt` | Sect. 4.6, Table 5 | ~1 h |
| 2 | `study_synthetic('B')` | `step4_mismatch.csv`, `step5_cv.csv`, `step6_projection.csv`, `step7_direct.csv`, `step8_capacity.csv`, `step10_weakform_cases.csv`, constant-speed check in `summary_B.txt` | Tables 6-7, Sect. 4.6, 4.9 | ~1 h |
| 3 | `study_synthetic('mc')` | `extra_mc.txt/.csv`: selection rates over 20 noise realizations | Table 3 | several hours |
| 4 | `study_synthetic('coef')` | `extra_coef.txt`: weak-form coefficients and bootstrap SEs; residual predictability | Table 4, Sect. 4.11 | ~15 min |
| 5 | `study_synthetic('noise')` | `extra_noise.txt/.csv`: selection versus noise level | Sect. 4.8 | ~10 min |
| 6 | `study_ekf('figure')` | `ekf_state_errors.txt`, `Fig_ekf_synthetic.pdf` | Fig. 4, Table 9 | ~5 min |
| 7 | `study_ekf('robustness')` | `ekf_robustness_*.csv` (+ LaTeX rows printed) | Tables 10-11 | ~10 min |
| 8 | `study_rig('bivariate')` | `rig_bivariate.txt`: speed check, bivariate weak form, 9 tracking settings, synthetic speed-error control | Sect. 5.2-5.3, Table 12 | ~10 min |
| 9 | `study_rig('figure_bivariate')` | `Fig_rig_simultaneous.pdf` | Fig. 6 | ~1 min |
| 10 | `study_rig('controls')` | `single_channel_controls.txt` | Sect. 5.4 | ~1 min |
| 11 | `study_rig('identify')` | `rig_single_channel_fitX.txt`, `..._fitY.txt` (+ .csv) | Table 13 | ~3 min |
| 12 | `study_rig('figure')` | `Fig_rig_single_channel.pdf` (single-sensor runs; not in the paper) | – | ~1 min |
| – | `study_rig('audit', <folder>)` | `rig_audit.txt`: optional consistency check of the rig data files against the raw sensor exports (requires the raw files) | – | ~1 min |

Copy `results/Fig_ekf_synthetic.pdf` and `results/Fig_rig_simultaneous.pdf`
into the manuscript folder to update the paper's figures.

### Which file produces which table or figure

| Paper item | Result file | Produced by |
|---|---|---|
| Table 1 (library) | – | `build_library.m` |
| Table 2 (benchmark) | – | `spiral_config.m` |
| Sect. 4.6 conditioning (kappa, mu_Phi) | `step1_conditioning_x/y.csv` | `study_synthetic('A')`, `diagnostics('conditioning')` |
| Sect. 4.6 constant-speed rank check | `summary_B.txt`, step [9] | `study_synthetic('B')` |
| Table 3 (selection rates) | `extra_mc.csv` | `study_synthetic('mc')` |
| Table 4 (coefficients) | `extra_coef.txt` | `study_synthetic('coef')` |
| Table 5 (ablations, HAVOK) | `step2_ablations.csv`, `step5_cv.csv`, `step6_projection.csv` | `study_synthetic('A','B')` |
| Table 6 (structural mismatch) | `step10_weakform_cases.csv` | `study_synthetic('B')` |
| Table 7 (library capacity) | `step8_capacity.csv` | `study_synthetic('B')` |
| Sect. 4.8 noise sweep | `extra_noise.csv` | `study_synthetic('noise')` |
| Sect. 4.11 residual predictability | `extra_coef.txt` (second part) | `study_synthetic('coef')` |
| Fig. 4, Table 9 (EKF) | `Fig_ekf_synthetic.pdf`, `ekf_state_errors.txt` | `study_ekf('figure')` |
| Tables 10-11 (EKF robustness) | `ekf_robustness_mismatch.csv`, `..._noise.csv` | `study_ekf('robustness')` |
| Sect. 5.2 speed check (tracked vs constant deceleration) | `rig_bivariate.txt`, part (1) | `study_rig('bivariate')` |
| Table 12 (simultaneous record) | `rig_bivariate.txt`, parts (2)-(4) | `study_rig('bivariate')` |
| Sect. 5.3 synthetic speed-error control | `rig_bivariate.txt`, part (5) | `study_rig('bivariate')` |
| Fig. 6 (rig) | `Fig_rig_simultaneous.pdf` | `study_rig('figure_bivariate')` |
| Table 13 (single-sensor runs) | `rig_single_channel_fitX/Y.txt` | `study_rig('identify')` |
| Sect. 5.4 synthetic calibration | `single_channel_controls.txt` | `study_rig('controls')` |

---

## 4. How the method is implemented

### 4.1 Data flow

```
spiral_config ──> simulate_rotor ──> record D (t, x, y, xd, yd, xdd, ydd, Omega, Omegadot)
                      │  rotor_rhs (full 4-DOF model; alpha, beta never leave this step)
                      │  sg_derivative (Savitzky-Golay, window from {21, 31, 51})
                      v
            build_library  (seed + 8 candidates per equation)
                      v
            weak_form      (identification: weak form + Bonferroni)  ──> selected terms
                      v
            diagnostics    (held-out gain, k-step prediction, conditioning,
                            residual predictability, library capacity)

regression_baselines : pointwise OLS / spaced OLS / block bootstrap   (comparison, Table 4 SEs)
spiral_po_loop       : gated network formulation                      (ablations, Table 5)
havok_baseline       : delay-coordinate model                         (Table 5)
study_ekf            : EKF given the 4-DOF model                      (Fig. 4, Tables 9-11)
study_rig            : estimate_speed + weak_form on the rig records  (Sect. 5)
```

### 4.2 The record `D`

Every function takes a record in the format returned by `simulate_rotor`:

| Field | Meaning |
|---|---|
| `t` | time [s], N x 1 |
| `x`, `y` | measured displacements [m] |
| `xd`, `yd`, `xdd`, `ydd` | Savitzky-Golay derivatives |
| `win_x`, `win_y` | SG window lengths (samples) |
| `Omega`, `Omegadot` | spin speed [rad/s] and its rate (known input) |
| `x_true`, `y_true`, `traj` | noise-free displacements and the full true state (synthetic only) |
| `params`, `cfg` | parameters and configuration |

A single-channel rig record (`study_rig`) has `t`, `x` (the channel), `xd`,
`xdd`, `Omega`, `Omegadot`.

### 4.3 Seed and library (`build_library.m`)

For the x-equation (the y-equation is obtained by exchanging x and y):

* seed, always fitted: `Omega^2 cos(theta)`, `Omega^2 sin(theta)`, `-xd`, `-x`,
  with `theta = ∫ Omega dt`;
* candidates: `Omega*ydot` (gyroscopic), `y` (cross-channel),
  `(x^2+y^2)*x` (isotropic cubic), and the competing forms `(x^2+y^2)*xdot`,
  `x^3`, `x^2*y`, `Omega^2*x`, `Omegadot*x`.

### 4.4 Identification (`weak_form.m`)

The acceleration equation is multiplied by the test functions
`psi_m = (1 - tau^2)^4` (half-width h = 25 ms, disjoint supports; about 160 of
them on the 8 s record) and integrated. Integration by parts moves the
derivatives onto `psi`, so the acceleration is never differentiated numerically:

```
∫ psi xdd = ∫ psi'' x,    ∫ psi xd = -∫ psi' x,    ∫ psi Omega yd = -∫ (psi' Omega + psi Omegadot) y
```

With white measurement noise, the measurement-noise part of the errors is
independent across test functions, so classical OLS standard errors apply to
it. The part of the dynamics the library does not capture is correlated across
neighboring test functions; at low noise, where it dominates, the standard
errors are optimistic (paper, Sect. 2.2.4 and 4.8). A candidate is
admitted when |t| > z_{1-alpha/(2K)} (Bonferroni over the K candidates; 2.73
for K = 8, 2.50 for K = 4). Candidates that are linear combinations of the seed
columns are removed first; for example, `Omegadot*x` is a multiple of `x` under
a constant deceleration.

Why not pointwise regression? Savitzky-Golay derivative errors form a moving
average of order (window - 1), so the effective sample size is about N/window.
Pointwise OLS with classical standard errors treats about 16,000 correlated
samples as independent and admits absent mechanisms (`regression_baselines`,
Table 3).

### 4.5 Diagnostics (`diagnostics.m`)

* `'heldout'`: seed and closure are refitted on the identification record and
  applied unchanged to an independent record (noise seed 77, unbalance phase
  +0.7 rad). The gain is the share of the held-out post-seed residual that the
  closure explains.
* `'predict'`: 10 ms and 50 ms predictions of x and y from the measured state.
* `'conditioning'`: singular values, kappa and mu_Phi of the column-normalized
  signature matrix.
* `'predictability'`: Step 7. Can a network still predict the residual from
  xi? The test is a one-sided t-test of the per-chunk log MSE ratio.
* `'capacity'`: the share of the noise-free closure that the library can
  represent, which gives epsilon_0.

### 4.6 Gated network formulation (`spiral_po_loop.m`)

This is the network-based variant of the procedure (paper, Sect. 2.2.9):
residual network, projection on Halton points, Gate 1 t-screen, clustering,
and Gate 2 cross-validated F-test. `'preset','pointwise'` runs it with pointwise
tests and amplitude-stratified folds; the default `'valid'` preset uses
effective-sample-size corrections and time-blocked validation. Each choice is
also available as a separate option. It is used only for the ablations
(Table 5) and the "Gated network" rows of Table 3.

### 4.7 Hidden-state reconstruction (`study_ekf.m`)

This is an 8-state EKF with the 4-DOF model as process model and {x, y} as
measurement. It is a model-based step: the filter is *given* the hidden-subsystem
equations, and it is not part of the identification.

### 4.8 Rig (`study_rig.m`)

The spin speed was not recorded on the rig, so it is estimated by order
tracking of the response (`estimate_speed.m`, 1.95 Hz bins, step-center
interpolation).

* Simultaneous record (`'bivariate'`): x and y from two sensors at 1 kHz.
  The speed column supplied with this record assumes a constant deceleration.
  Part (1) shows that the response and its second harmonic follow the tracked
  speed instead: the line is up to 5.9 Hz off. The bivariate weak form is run
  with the tracked speed for nine tracking settings and with the line. Part (5)
  gives the synthetic benchmark a line that is 4 Hz off; it then admits
  gyroscopic terms that are absent from the data.
* Single-sensor runs (`'identify'`): one direction per run, so only
  {s^3, s^2 sd, Omega^2 s, Omegadot s} are tested. `'controls'` applies the
  same analysis to the synthetic x channel.

---

## 5. Settings used in the paper

| Setting | Value | Where |
|---|---|---|
| Benchmark | m = 1 kg, kL = 1000 N/m, kNL = 5e6 N/m^3, cb = 1.4 N s/m per bearing, a/L = 0.3, e = 0.01 m, phi0 = 0.4 rad | `spiral_config.m` |
| Record | 8 s, 60 -> 5 Hz linear coast-down, RK4 at dt = 5e-4 s, 2 % noise | `spiral_config.m` |
| Noise seeds | 31, 41, 51 (identification); 101-120 (Monte Carlo); 201-205 (noise sweep); 77 + phase 0.7 rad (independent record) | `study_synthetic.m` |
| SG derivatives | order 4, window from {21, 31, 51} by hold-out CV | `sg_derivative.m` |
| Weak form | h = 0.025 s, psi = (1 - tau^2)^4, alpha = 0.05 Bonferroni | `weak_form.m` |
| Block bootstrap | B = 1000, block length 4 SG windows, seed 1 | `regression_baselines.m` |
| HAVOK | 50 delays per channel, stride 2 ms, rank 15 | `havok_baseline.m` |
| EKF | Q_scale = 1e-3, R from second differences | `study_ekf.m` |
| Rig, simultaneous | 1 kHz, tracked speed > 3 Hz (5.45 s), SG windows {11,21,31}, STFT 512 samples | `study_rig.m` |
| Rig, single-sensor | 4 kHz -> 2 kHz, 1-24 s, K = 4 candidates | `study_rig.m` |

## 6. Reference values for checking a run

| Quantity | Value |
|---|---|
| Weak form, nominal, x | Omega*ydot (t = -7.9), (x^2+y^2)x (t = -216) |
| Weak form, nominal, y | Omega*xdot (t = 3.5), (x^2+y^2)y (t = -284) |
| Held-out gain, weak form | 1.84 % (x), 1.42 % (y) |
| Conditioning, full library | kappa = 35 / 25, mu_Phi = 0.004 / 0.008 |
| Library capacity, nominal | R^2 = 0.41 (x), 0.65 (y) |
| EKF, independent record | alpha 1.41 %, beta 1.62 % relative RMSE |
| Rig, simultaneous record | Omega^2*x (t = -12.1), Omega^2*y (t = -6.5) only; 9/9 settings |
| Rig, single-sensor fit on X | Omega^2 s (t = -9.07), Omegadot s (t = 3.48); held-out gain on Y 14.2 % |

## 7. Reproducibility notes

* `results/` ships with the reference output of a complete verification run
  in Octave 8.4 (the simulated-record cache is not included; it is rebuilt on
  the first run). Running the package again should reproduce these files.

* All random numbers are seeded. In Octave every result is bit-for-bit
  reproducible. The EKF robustness study seeds with `rng` in MATLAB and
  `rand('state')` in Octave, so its noise realizations, and therefore the last
  digits of Tables 10-11, differ between the two. (The paper's Tables 10-11
  come from a MATLAB run; the shipped `ekf_robustness_*.csv` come from Octave,
  e.g. alpha RMSE 2.83e-4 instead of 2.85e-4 in the matched case.)

## 8. Data

* `Experimental_Rotor_Data_SPIRAL_PO.csv`: simultaneous x and y displacements
  of the laboratory rotor during a coast-down (two laser sensors, 1 kHz). The
  rig has no speed sensor; the speed column assumes a constant deceleration,
  and the paper uses a speed estimated from the response instead.
* `SIG_9_9_2022.mat`: single-sensor coast-downs from a separate session
  (4 kHz), one lateral direction per run (`SIG_CH_X_*`, `SIG_CH_Y_*`). The
  `SIG_CH_Z_*` channels measure the axial direction and are not used, since
  the model is planar.

## 9. How to cite

If you use this code, please cite the paper above. A BibTeX entry will be
added when the paper is published.

## 10. License

The code is released under the MIT License (see `LICENSE`).
