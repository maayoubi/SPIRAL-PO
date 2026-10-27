function out = study_ekf(part)
% STUDY_EKF  Model-based reconstruction of the hidden tilt angles with an
% extended Kalman filter given the 4-DOF model and the noisy x, y
% (Sect. 4.12-4.13 of the paper).
%
%   study_ekf                % both parts
%   study_ekf('figure')      % Fig. 4 and Table 9: independent record
%                            % (noise seed 77, unbalance phase +0.7 rad)
%   study_ekf('robustness')  % Tables 10-11: filter bearing parameters
%                            % perturbed by 0/5/10/20 %, noise 2-20 %
%
% Output (results/): ekf_state_errors.txt, Fig_ekf_synthetic.pdf/.png,
% ekf_robustness_mismatch.csv, ekf_robustness_noise.csv,
% ekf_robustness_results.mat. The robustness tables are also printed as
% LaTeX rows.

if nargin < 1, part = 'all'; end
RES = result_dir();
out = struct();
if any(strcmpi(part, {'all','figure'})), out.figure = ekf_figure(RES); end
if any(strcmpi(part, {'all','robustness'})), ekf_robustness(RES); end
end

% ========================================================================
function E = ekf_figure(RES)
cfg = spiral_config('nominal', 77);
cfg.params.phi0 = cfg.params.phi0 + 0.7;          % independent record
D = simulate_rotor(cfg);                          % D.traj = true state (hidden)

[z, ~] = ekf_4dof(D.t, D.x, D.y, D.params, D.Omega_fun, D.theta_fun, 'Q_scale', 1e-3);

names = {'x','y','\alpha','\beta'};
units = {'m','m','rad','rad'};
E = struct('state', {{}}, 'rmse', [], 'rel', [], 'corr', []);
fid = fopen(fullfile(RES,'ekf_state_errors.txt'), 'w');
hdr = sprintf('%-6s %12s %10s %10s\n', 'state', 'RMSE', 'rel.err', 'corr');
fprintf('%s', hdr); fprintf(fid, '%s', hdr);
for k = 1:4
    tr = D.traj(:,k); es = z(:,k);
    rm = sqrt(mean((tr - es).^2));
    rel = rm / std(tr);
    c = corrcoef(tr, es); c = c(1,2);
    E.state{k} = names{k}; E.rmse(k) = rm; E.rel(k) = rel; E.corr(k) = c;
    line = sprintf('%-6s %12.3e %9.2f%% %10.5f  [%s]\n', strrep(names{k},'\',''), rm, 100*rel, c, units{k});
    fprintf('%s', line); fprintf(fid, '%s', line);
end
fclose(fid);

% ---------------- figure: full record (left) and zoom (right) ----------------
blue = [0.13 0.27 0.60]; red = [0.80 0.15 0.15];
ylab = {'x [mm]', 'y [mm]', '\alpha [mrad]', '\beta [mrad]'};
zoom = D.t >= 4.0 & D.t <= 4.3;
fig = figure('Units', 'centimeters', 'Position', [2 2 18 16], 'Color', 'w');
for k = 1:4
    tr = 1e3*D.traj(:,k); es = 1e3*z(:,k);
    subplot(4, 2, 2*k-1);
    plot(D.t, tr, '-', 'Color', blue, 'LineWidth', 0.8); hold on
    plot(D.t, es, '--', 'Color', red, 'LineWidth', 0.8);
    ylabel(ylab{k}); xlim([D.t(1) D.t(end)]); box on
    if k == 1, legend({'true', 'EKF estimate'}, 'Location', 'northeast', 'Box', 'off'); end
    if k == 4, xlabel('time [s]'); end
    subplot(4, 2, 2*k);
    plot(D.t(zoom), tr(zoom), '-', 'Color', blue, 'LineWidth', 1.0); hold on
    plot(D.t(zoom), es(zoom), '--', 'Color', red, 'LineWidth', 1.0);
    xlim([4.0 4.3]); box on
    if k == 4, xlabel('time [s]'); end
end
set(findall(fig, '-property', 'FontSize'), 'FontSize', 9);
print(fig, fullfile(RES,'Fig_ekf_synthetic.pdf'), '-dpdf', '-painters', '-bestfit');
print(fig, fullfile(RES,'Fig_ekf_synthetic.png'), '-dpng', '-r200');
end

% ========================================================================
function ekf_robustness(RES)
% Parameter-mismatch and noise sweeps on an independent record.
    p0 = base_params();              % true, data-generating parameters
    mismatch_levels = [0.00 0.05 0.10 0.20];
    noise_levels    = [0.02 0.05 0.10 0.20];
    % independent record (noise seed 77, phase +0.7 rad)
    val_seed   = 20260624;           % NOT the seed used for model discovery
    p0.phi0    = p0.phi0 + 0.7;      % different unbalance phase than discovery
    % ---------- Study 1: parameter mismatch (noise fixed at 2%) ----------
    fprintf('\n%% ---- Table tab:ekf_mismatch (noise = 2%%) ----\n');
    R1 = zeros(numel(mismatch_levels),6);
    for i = 1:numel(mismatch_levels)
        d = mismatch_levels(i);
        [tt,q_true]  = simulate_truth(p0, 0.02, val_seed);   % truth + 2% noise obs
        yobs         = q_true(:,1:2) + meas_noise(q_true(:,1:2), 0.02, val_seed+1);
        p_ekf        = perturb_bearings(p0, d);              % mismatched filter model
        zhat         = run_ekf(tt, yobs, p_ekf);
        R1(i,:)      = metrics_row(q_true, zhat);
        emit_row(sprintf('$%g\\%%$', 100*d), R1(i,:), false);
    end
    % ---------- Study 2: noise sweep (parameters matched) ----------
    fprintf('\n%% ---- Table tab:ekf_noise (delta = 0) ----\n');
    R2 = zeros(numel(noise_levels),6);
    for i = 1:numel(noise_levels)
        s = noise_levels(i);
        [tt,q_true]  = simulate_truth(p0, s, val_seed);
        yobs         = q_true(:,1:2) + meas_noise(q_true(:,1:2), s, val_seed+1);
        zhat         = run_ekf(tt, yobs, p0);               % matched model
        R2(i,:)      = metrics_row(q_true, zhat);
        win          = stable_window(tt, q_true, zhat);     % seconds before divergence
        emit_row(sprintf('$%g\\%%$', 100*s), R2(i,1:4), true, win);
    end
    save(fullfile(RES,'ekf_robustness_results.mat'),'R1','R2','mismatch_levels','noise_levels');
    csvwrite_safe(fullfile(RES,'ekf_robustness_mismatch.csv'), [mismatch_levels(:) R1]);
    csvwrite_safe(fullfile(RES,'ekf_robustness_noise.csv'),    [noise_levels(:)    R2]);
    fprintf('\nSaved ekf_robustness_results.mat and CSVs.\n');
end

function [tt, q_true] = simulate_truth(p, noise_std_frac, seed) %#ok<INUSD>
% Integrate the full 4-DOF rotor and return the TRUE (clean) states.
%   tt     : (N x 1) time vector
%   q_true : (N x 4) true states [x y alpha beta]  (noise added by caller)
    set_seed(seed);
    params = ekf_params_from(p);
    [Omega_fun, theta_fun] = speed_schedule(p);
    tt = (0:p.dt:p.T_end).';
    N  = numel(tt);
    z    = zeros(8,1);                 % [x y alpha beta xd yd ad bd]
    traj = zeros(N,8);  traj(1,:) = z.';
    dt = p.dt;
    for i = 1:N-1
        ti = tt(i);
        k1 = rotor_rhs(ti,        z,           params, Omega_fun, theta_fun);
        k2 = rotor_rhs(ti+dt/2,   z+dt/2*k1,   params, Omega_fun, theta_fun);
        k3 = rotor_rhs(ti+dt/2,   z+dt/2*k2,   params, Omega_fun, theta_fun);
        k4 = rotor_rhs(ti+dt,     z+dt*k3,     params, Omega_fun, theta_fun);
        z  = z + dt/6*(k1+2*k2+2*k3+k4);
        traj(i+1,:) = z.';
    end
    q_true = traj(:,1:4);
end
% =========================================================================
function zhat = run_ekf(tt, yobs, p_ekf)
% Run the 8-state EKF on (x,y) observations with filter model p_ekf.
%   zhat : (N x 8) estimate [x y alpha beta xd yd ad bd]
    params = ekf_params_from(p_ekf);
    [Omega_fun, theta_fun] = speed_schedule(p_ekf);
    [z_est, ~] = ekf_4dof(tt, yobs(:,1), yobs(:,2), params, ...
                          Omega_fun, theta_fun, 'Q_scale', 1e-3);
    zhat = z_est;
end
% =========================================================================
function params = ekf_params_from(p)
% Map the scalar-bearing base_params struct onto the per-bearing struct that
% rotor_rhs.m / ekf_4dof expect.
    xib = 1 - p.xi;
    L1 = [ xib, 0,   0,             p.xi*xib*p.L ;
           0,   xib, -p.xi*xib*p.L, 0            ];
    L2 = [ p.xi, 0,    0,            -p.xi*xib*p.L ;
           0,    p.xi, p.xi*xib*p.L, 0             ];
    params.m   = p.m;   params.Id = p.Id;  params.Ip = p.Ip;
    params.kL1 = p.kL;  params.kL2  = p.kL;
    params.kNL1= p.kNL; params.kNL2 = p.kNL;
    params.cb1 = p.cb;  params.cb2  = p.cb;
    params.e   = p.e;   params.phi0 = p.phi0;
    params.L1  = L1;    params.L2   = L2;
end
function [Omega_fun, theta_fun] = speed_schedule(p)
% Prescribed run-down schedule 60 Hz -> 5 Hz over T_end (as spiral_config).
    f0 = 60; f1 = 5; Tend = p.T_end; decel = (f0-f1)/Tend;
    Omega_fun = @(tt) 2*pi*max(f1, f0 - decel*tt);
    theta_fun = @(tt) 2*pi*(f0.*tt - 0.5*decel.*min(tt,Tend).^2);
end
% =========================================================================
% Parameters and perturbation
% =========================================================================
function p = base_params()
    p.m   = 1.0;            p.R = 0.05;
    p.Ip  = 0.5*p.m*p.R^2;  p.Id = 0.5*p.Ip;
    p.a   = 0.3;            p.L = 1.0;   p.xi = p.a/p.L;
    p.kL  = 1000;           p.kNL = 5e6; p.cb = 1.4;   % bearing params (perturbed)
    p.e   = 0.01;           p.phi0 = 0.4;
    p.dt  = 5e-4;           p.T_end = 8.0;             % 60 -> 5 Hz run-down
end
function p = perturb_bearings(p, delta)
% Joint relative mismatch applied to the filter's bearing model.
    p.kL  = p.kL  * (1 + delta);
    p.kNL = p.kNL * (1 + delta);
    p.cb  = p.cb  * (1 + delta);
end
% =========================================================================
% Noise, metrics, helpers
% =========================================================================
function n = meas_noise(sig, frac, seed)
    set_seed(seed);
    s = std(sig,0,1);                       % per-channel std
    n = randn(size(sig)) .* (frac*s);
end
function row = metrics_row(q_true, zhat)
% Returns [aRMSE aCORR bRMSE bCORR xyRMSE unused] (6 cols).
    a_t = q_true(:,3);  b_t = q_true(:,4);
    a_h = zhat(:,3);    b_h = zhat(:,4);
    x_t = q_true(:,1);  y_t = q_true(:,2);
    x_h = zhat(:,1);    y_h = zhat(:,2);
    row = [ rmse(a_h,a_t)  pcorr(a_h,a_t) ...
            rmse(b_h,b_t)  pcorr(b_h,b_t) ...
            0.5*(rmse(x_h,x_t)+rmse(y_h,y_t))  0 ];
end
function r = rmse(a,b),  r = sqrt(mean((a(:)-b(:)).^2)); end
function c = pcorr(a,b)
    a=a(:)-mean(a(:)); b=b(:)-mean(b(:));
    c = (a'*b) / max(sqrt((a'*a)*(b'*b)), eps);
end
function w = stable_window(tt, q_true, zhat)
% Longest leading segment (seconds) before reconstruction error blows up.
    err = sqrt(sum((zhat(:,1:4)-q_true(:,1:4)).^2, 2));
    thr = 5 * median(err);                  % blow-up = 5x median error
    k = find(err > thr, 1, 'first');
    if isempty(k), w = tt(end)-tt(1); else, w = tt(k)-tt(1); end
end
function emit_row(label, vals, is_noise, win)
% Print a LaTeX table row matching the manuscript tables.
    if nargin < 4, win = NaN; end
    if is_noise
        fprintf('%s & %.2g & %.4f & %.2g & %.4f & %.2f\\,s \\\\\n', ...
                label, vals(1), vals(2), vals(3), vals(4), win);
    else
        fprintf('%s & %.2g & %.4f & %.2g & %.4f & %.2g \\\\\n', ...
                label, vals(1), vals(2), vals(3), vals(4), vals(5));
    end
end
function set_seed(seed)
    try, rng(seed);                      % MATLAB
    catch, rand('state',seed); randn('state',seed); end  % Octave fallback
end
function csvwrite_safe(fname, M)
    try, csvwrite(fname, M);
    catch, dlmwrite(fname, M, ','); end
end

% ========================================================================
function [z_est, P_diag] = ekf_4dof(t, x_meas, y_meas, params, Omega_fun, theta_fun, varargin)
% Extended Kalman filter for the 8-state rotor z = [x y alpha beta xd yd
% alphad betad] with the full nonlinear 4-DOF model (rotor_rhs.m) as the
% process model and {x, y} as the only measurement (Sect. 4.12, Table 8).
%   mean: RK4 propagation; covariance: Psi = I + dt F + dt^2 F^2 / 2 with a
%   finite-difference Jacobian F; Q = Q_scale diag(1e-2*ones(4), ones(4));
%   R estimated from the second differences of x, y unless 'R_meas' is
%   given; P0 = diag(1e-6,1e-6, P0s,P0s, 1e-4,1e-4, 10 P0s,10 P0s).
% Options: 'Q_scale' (1e-3), 'R_meas' ([]), 'z0' ([x1 y1 0...0]), 'P0_scale' (1e-4)
p = inputParser;
p.addParameter('Q_scale', 1e-3);
p.addParameter('R_meas', []);
p.addParameter('z0', []);
p.addParameter('P0_scale', 1e-4);
p.parse(varargin{:});
o = p.Results;

N = numel(t);
dt = t(2) - t(1);
% the filter uses the plain 4-DOF model (extended-model fields removed)
params = strip_extended(params);

if isempty(o.R_meas)
    % noise estimate from second differences
    d2x = diff(x_meas,2); d2y = diff(y_meas,2);
    Rx = var(d2x)/6; Ry = var(d2y)/6;   % 2nd-diff of white noise has variance 6*sigma^2
    R = diag([Rx, Ry]);
else
    if isscalar(o.R_meas)
        R = o.R_meas * eye(2);
    else
        R = diag(o.R_meas(:));
    end
end

if isempty(o.z0)
    z0 = [x_meas(1); y_meas(1); 0; 0; 0; 0; 0; 0];
else
    z0 = o.z0;
end

% process noise: larger on the velocity states
Q = o.Q_scale * diag([1e-2,1e-2,1e-2,1e-2, 1,1,1,1]);

P0 = diag([1e-6,1e-6, o.P0_scale,o.P0_scale, 1e-4,1e-4, o.P0_scale*10,o.P0_scale*10]);

H = zeros(2,8); H(1,1)=1; H(2,2)=1;

z_est = zeros(N,8);
P_diag = zeros(N,8);
z = z0; P = P0;
z_est(1,:) = z'; P_diag(1,:) = diag(P)';

for k = 1:N-1
    tk = t(k);

    % --- predict: RK4 mean, finite-difference Jacobian for the covariance ---
    f = @(tt,zz) rotor_rhs(tt, zz, params, Omega_fun, theta_fun);
    k1 = f(tk, z);
    k2 = f(tk+dt/2, z+dt/2*k1);
    k3 = f(tk+dt/2, z+dt/2*k2);
    k4 = f(tk+dt, z+dt*k3);
    z_pred = z + dt/6*(k1+2*k2+2*k3+k4);

    Fjac = numerical_jacobian_local(f, tk, z);
    Phi = eye(8) + dt*Fjac + 0.5*dt^2*(Fjac*Fjac);   % 2nd-order discretization
    P_pred = Phi*P*Phi' + Q;

    % --- update ---
    zmeas = [x_meas(k+1); y_meas(k+1)];
    yres = zmeas - H*z_pred;
    S = H*P_pred*H' + R;
    K = (P_pred*H') / S;
    z = z_pred + K*yres;
    P = (eye(8) - K*H)*P_pred;
    P = 0.5*(P+P');   % enforce symmetry against roundoff

    z_est(k+1,:) = z';
    P_diag(k+1,:) = diag(P)';
end

end


function Fjac = numerical_jacobian_local(f, tt, z)
n = numel(z);
Fjac = zeros(n,n);
f0 = f(tt,z);
heps = 1e-6 * max(abs(z), 1e-6);
for j = 1:n
    zp = z; zp(j) = zp(j) + heps(j);
    Fjac(:,j) = (f(tt,zp) - f0) / heps(j);
end
end

function P = strip_extended(P)
for f = {'aniso','gyro_scale','bw'}
    if isfield(P, f{1}), P = rmfield(P, f{1}); end
end
end

function d = result_dir()
d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results');
if ~exist(d, 'dir'), mkdir(d); end
end
