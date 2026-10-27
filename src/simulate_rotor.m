function D = simulate_rotor(cfg, varargin)
% SIMULATE_ROTOR  Synthetic measurement record from the 4-DOF rotor:
% integrate the full model (RK4), keep only x and y, add measurement noise
% and compute the Savitzky-Golay derivatives.
%
%   D = simulate_rotor(spiral_config('nominal', 31))
%   D = simulate_rotor(D, 'renoise', seed, noise_frac)
%
% The second form draws new measurement noise for an existing record.
%
% Output struct D
%   t                  time [s] (N x 1)
%   x, y               measured lateral displacements [m]
%   xd, yd, xdd, ydd   Savitzky-Golay derivative estimates
%   win_x, win_y       SG windows (chosen from [21 31 51]), sg_order = 4
%   Omega, Omegadot    spin speed [rad/s] and its rate (known input)
%   x_true, y_true     noise-free displacements (synthetic only)
%   traj               true full state, N x 8 (N x 10 with hysteresis)
%   Omega_fun, theta_fun  spin schedule as function handles
%   params, cfg        parameters and configuration used

if nargin >= 2 && ischar(varargin{1}) && strcmpi(varargin{1}, 'renoise')
    D = renoise(cfg, varargin{2}, varargin{3});
    return;
end

P  = cfg.params;
dt = cfg.dt; N = round(cfg.Tend/dt) + 1;
t  = (0:N-1)'*dt;
f0 = cfg.f_start; f1 = cfg.f_end; T = cfg.Tend;
rate = (f1 - f0)/T;
lo = min(f0,f1); hi = max(f0,f1);
Omega_fun = @(tt) 2*pi*min(hi, max(lo, f0 + rate*tt));
theta_fun = @(tt) 2*pi*(f0.*tt + 0.5*rate.*min(tt,T).^2);

% --- integrate the full model (fixed-step RK4, n_sub sub-steps/sample) ---
nz = 8 + 2*P.bw.on;
z = zeros(nz,1);
traj = zeros(N,nz);
f = @(tt,zz) rotor_rhs(tt, zz, P, Omega_fun, theta_fun);
ns = 1; if isfield(cfg,'n_sub'), ns = cfg.n_sub; end
h = dt/ns;
for i = 1:N-1
    ti = t(i);
    for j = 1:ns
        tj = ti + (j-1)*h;
        k1 = f(tj, z);
        k2 = f(tj+h/2, z+h/2*k1);
        k3 = f(tj+h/2, z+h/2*k2);
        k4 = f(tj+h,   z+h*k3);
        z = z + h/6*(k1+2*k2+2*k3+k4);
    end
    traj(i+1,:) = z';
end
if ~all(isfinite(traj(:)))
    error('simulate_rotor:diverged', 'integration diverged for case %s; increase cfg.n_sub', cfg.case);
end

% --- spin speed (lightly smoothed) ---
Om_true = Omega_fun(t);
[Omega, Omegadot] = smooth_speed(t, t(1:50:end), Om_true(1:50:end)/(2*pi), 0.3);

D.t = t;
D.x_true = traj(:,1); D.y_true = traj(:,2);
D.Omega = Omega; D.Omegadot = Omegadot;
D.traj = traj;
D.Omega_fun = Omega_fun; D.theta_fun = theta_fun;
D.params = P; D.cfg = cfg;
D = renoise(D, cfg.noise_seed, cfg.noise_frac);
end

% -------------------------------------------------------------------------
function D = renoise(D, seed, noise_frac)
% White noise at noise_frac of each channel's standard deviation, then SG
% derivatives with a cross-validated window.
dt = D.t(2) - D.t(1); N = numel(D.t);
rand('seed', seed); randn('seed', seed);
D.x = D.x_true + noise_frac*std(D.x_true)*randn(N,1);
D.y = D.y_true + noise_frac*std(D.y_true)*randn(N,1);
[D.xd, D.xdd, D.win_x] = sg_derivative(D.x, dt, [21 31 51], 4);
[D.yd, D.ydd, D.win_y] = sg_derivative(D.y, dt, [21 31 51], 4);
D.sg_order = 4;
D.cfg.noise_seed = seed; D.cfg.noise_frac = noise_frac;
end

function [Omega_s, Omegadot_s] = smooth_speed(t, t_frame, f_frame, window_s)
% Moving-average smoothing of a frame-rate speed curve f_frame(t_frame)
% [Hz], interpolation to t, central-difference rate.
frame_dt = mean(diff(t_frame));
win = max(3, round(window_s / frame_dt));
if mod(win,2) == 0, win = win+1; end
f_frame = f_frame(:); n = numel(f_frame); hw = floor(win/2);
f_s = zeros(n,1);
for k = 1:n
    f_s(k) = mean(f_frame(max(1,k-hw):min(n,k+hw)));
end
f_full = max(interp1(t_frame, f_s, t, 'linear', 'extrap'), 0);
Omega_s = 2*pi*f_full;
dt = t(2)-t(1);
Omegadot_s = zeros(size(Omega_s));
Omegadot_s(2:end-1) = (Omega_s(3:end) - Omega_s(1:end-2)) / (2*dt);
Omegadot_s(1) = (Omega_s(2)-Omega_s(1))/dt;
Omegadot_s(end) = (Omega_s(end)-Omega_s(end-1))/dt;
end
