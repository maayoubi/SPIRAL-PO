function cfg = spiral_config(case_name, noise_seed)
% SPIRAL_CONFIG  Synthetic benchmark settings (Table 2 of the paper).
%
%   cfg = spiral_config()                  % nominal benchmark, noise seed 31
%   cfg = spiral_config('hysteresis', 7)   % a structural-mismatch case
%   D   = simulate_rotor(cfg)              % integrate and measure
%
% Cases
%   'nominal'     isotropic bearings, cubic bearing stiffness, gyroscopic
%                 coupling
%   'no_duffing'  knock-out control: cubic bearing stiffness removed
%   'no_gyro'     knock-out control: gyroscopic matrix removed
%   'aniso'       anisotropic bearings (kx = 1.3 kL, ky = 0.7 kL)
%   'hysteresis'  Bouc-Wen hysteretic element at bearing 1
%   'params'      all physical parameters perturbed by up to +/-20 %
%   'runup'       run-up 5 -> 60 Hz instead of a coast-down
%
% Common settings: 8 s record, linear 60 -> 5 Hz coast-down (except
% 'runup'), RK4 at dt = 5e-4 s (2 kHz), 2 % white measurement noise on x
% and y (relative to each channel's standard deviation).
%
% Output fields
%   cfg.case, cfg.noise_seed, cfg.dt, cfg.Tend, cfg.f_start, cfg.f_end,
%   cfg.noise_frac, cfg.n_sub (RK4 sub-steps per sample)
%   cfg.params   physical parameters (struct read by rotor_rhs.m):
%                m, Id, Ip, L1, L2, kL1, kL2, kNL1, kNL2, cb1, cb2, e, phi0,
%                aniso (2x2), gyro_scale, bw (Bouc-Wen settings)
%   cfg.spiral   settings of the gated network formulation (spiral_po_loop.m)

if nargin < 1 || isempty(case_name), case_name = 'nominal'; end
if nargin < 2 || isempty(noise_seed), noise_seed = 31; end

cfg.case = case_name;
cfg.noise_seed = noise_seed;

% --- time base and speed schedule ---
cfg.dt   = 5e-4;
cfg.Tend = 8;
cfg.f_start = 60;   % Hz
cfg.f_end   = 5;    % Hz
cfg.noise_frac = 0.02;
cfg.n_sub = 1;          % RK4 sub-steps per output sample

% --- geometry (Table 1) ---
m = 1.0; R = 0.05;
Ip = 0.5*m*R^2; Id = 0.5*Ip;
a = 0.3; Lspan = 1.0; lam = a/Lspan; lamb = 1 - lam;   % lambda = a/L
P.m = m; P.Id = Id; P.Ip = Ip;
P.L1 = [lamb,0,0,lam*lamb*Lspan; 0,lamb,-lam*lamb*Lspan,0];
P.L2 = [lam,0,0,-lam*lamb*Lspan; 0,lam,lam*lamb*Lspan,0];
P.kL1 = 1000; P.kL2 = 1000;          % linear bearing stiffness [N/m]
P.kNL1 = 5e6; P.kNL2 = 5e6;          % cubic bearing stiffness [N/m^3]
P.cb1 = 1.4;  P.cb2 = 1.4;           % bearing damping [N s/m]
P.e = 0.01;   P.phi0 = 0.4;          % unbalance eccentricity [m], phase [rad]

% extended-model fields (defaults give the plain 4-DOF model)
P.aniso = [1 1; 1 1];                % [kx/kL, ky/kL] for bearing 1 (row 1), 2 (row 2)
P.gyro_scale = 1;
P.bw.on = false;
P.bw.kh = 0; P.bw.A = 1; P.bw.beta = 0; P.bw.gamma = 0; P.bw.n = 1;

switch lower(case_name)
    case 'nominal'
    case 'no_duffing'
        P.kNL1 = 0; P.kNL2 = 0;
    case 'no_gyro'
        P.gyro_scale = 0;
    case 'aniso'
        P.aniso = [1.3 0.7; 1.3 0.7];
    case 'hysteresis'
        % Bouc-Wen element at bearing 1 in both lateral directions:
        %   F_h = kh*w,  wdot = A*udot - beta*|udot|*|w|^(n-1)*w - gamma*udot*|w|^n
        % with u the bearing-1 displacement (L1*q). Saturation level
        % w_max = (A/(beta+gamma))^(1/n) = 0.02 m, kh = 0.5*kL1.
        P.bw.on = true; P.bw.kh = 0.5*P.kL1; P.bw.A = 1; P.bw.n = 1;
        P.bw.beta = 25; P.bw.gamma = 25;
    case 'params'
        % nine fixed factors in [0.8, 1.2]
        f = [1.12 0.86 1.17 0.91 1.08 0.83 1.19 0.88 1.05];
        P.m = P.m*f(1); P.Ip = P.Ip*f(2); P.Id = P.Id*f(3);
        P.kL1 = P.kL1*f(4); P.kL2 = P.kL2*f(5);
        P.kNL1 = P.kNL1*f(6); P.kNL2 = P.kNL2*f(7);
        P.cb1 = P.cb1*f(8); P.cb2 = P.cb2*f(9);
    case 'runup'
        % run-up; 4 RK4 sub-steps per sample
        cfg.f_start = 5; cfg.f_end = 60; cfg.n_sub = 4;
    otherwise
        error('spiral_config:badCase', 'unknown case "%s"', case_name);
end
cfg.params = P;

% --- default settings of spiral_po_loop.m ---
cfg.spiral.NH        = 20000;
cfg.spiral.max_iter  = 6;
cfg.spiral.nn_hidden = [12 12];
cfg.spiral.alpha     = 0.05;
end

