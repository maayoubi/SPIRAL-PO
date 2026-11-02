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

[z, ~] = ekf_4dof(D.t, D.x, D.y, D.params, D.Omega_fun, D.theta_fun, 'Q_scale', 1e-4);

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
zoom = D.t >= 9.0 & D.t <= 10.0;
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
    xlim([9.0 10.0]); box on
    if k == 4, xlabel('time [s]'); end
end
set(findall(fig, '-property', 'FontSize'), 'FontSize', 9);
print(fig, fullfile(RES,'Fig_ekf_synthetic.pdf'), '-dpdf', '-painters', '-bestfit');
print(fig, fullfile(RES,'Fig_ekf_synthetic.png'), '-dpng', '-r200');
end

% ========================================================================
function ekf_robustness(RES)
% Parameter-mismatch and noise sweeps on an independent record.
    cfg0 = spiral_config('nominal', 77);
    cfg0.params.phi0 = cfg0.params.phi0 + 0.7;     % independent record
    Dt = simulate_rotor(cfg0);                     % true trajectory
    tt = Dt.t; q_true = Dt.traj(:,1:4);
    p0 = Dt.params;
    mismatch_levels = [0.00 0.05 0.10 0.20];
    noise_levels    = [0.02 0.05 0.10 0.20];
    val_seed = 20260624;
    % ---------- Study 1: parameter mismatch (noise fixed at 2%) ----------
    fprintf('\n%% ---- Table tab:ekf_mismatch (noise = 2%%) ----\n');
    R1 = zeros(numel(mismatch_levels),6);
    for i = 1:numel(mismatch_levels)
        d = mismatch_levels(i);
        yobs  = q_true(:,1:2) + meas_noise(q_true(:,1:2), 0.02, val_seed+1);
        p_ekf = perturb_bearings(p0, d);                     % mismatched filter model
        zhat  = run_ekf(tt, yobs, p_ekf, Dt);
        R1(i,:) = metrics_row(q_true, zhat);
        emit_row(sprintf('$%g\\%%$', 100*d), R1(i,:), false);
    end
    % ---------- Study 2: noise sweep (parameters matched) ----------
    fprintf('\n%% ---- Table tab:ekf_noise (delta = 0) ----\n');
    R2 = zeros(numel(noise_levels),6);
    for i = 1:numel(noise_levels)
        s = noise_levels(i);
        yobs = q_true(:,1:2) + meas_noise(q_true(:,1:2), s, val_seed+1);
        zhat = run_ekf(tt, yobs, p0, Dt);                    % matched model
        R2(i,:) = metrics_row(q_true, zhat);
        win = stable_window(tt, q_true, zhat);
        emit_row(sprintf('$%g\\%%$', 100*s), R2(i,1:4), true, win);
    end
    save(fullfile(RES,'ekf_robustness_results.mat'),'R1','R2','mismatch_levels','noise_levels');
    csvwrite_safe(fullfile(RES,'ekf_robustness_mismatch.csv'), [mismatch_levels(:) R1]);
    csvwrite_safe(fullfile(RES,'ekf_robustness_noise.csv'),    [noise_levels(:)    R2]);
    fprintf('\nSaved ekf_robustness_results.mat and CSVs.\n');
end

function zhat = run_ekf(tt, yobs, p_ekf, Dt)
% 8-state EKF on (x, y) observations with filter model p_ekf.
    [z_est, ~] = ekf_4dof(tt, yobs(:,1), yobs(:,2), p_ekf, ...
                          Dt.Omega_fun, Dt.theta_fun, 'Q_scale', 1e-4);
    zhat = z_est;
end

function p = perturb_bearings(p, delta)
% Joint relative mismatch of the filter's bearing model (stiffness, cubic
% stiffness, cross-axis stiffness and damping).
    p.kL1 = p.kL1*(1+delta); p.kL2 = p.kL2*(1+delta);
    p.kxy = p.kxy*(1+delta); p.kyx = p.kyx*(1+delta);
    p.kNL1 = p.kNL1*(1+delta); p.kNL2 = p.kNL2*(1+delta);
    p.cb1 = p.cb1*(1+delta); p.cb2 = p.cb2*(1+delta);
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
%   mean: RK4 propagation; covariance: Phi = expm(dt F) with a
%   finite-difference Jacobian F. State scales s from the measured response
%   (std of x, y; tilts over a unit span; rates times the highest spin
%   speed): P0 = diag(s.^2), Q = Q_scale diag(s.^2). R estimated from the
%   second differences of x, y unless 'R_meas' is given.
% Options: 'Q_scale' (1e-4), 'R_meas' ([]), 'z0' ([x1 y1 0...0])
p = inputParser;
p.addParameter('Q_scale', 1e-4);
p.addParameter('R_meas', []);
p.addParameter('z0', []);
p.parse(varargin{:});
o = p.Results;

N = numel(t);
dt = t(2) - t(1);
% the filter uses the 4-DOF model without hysteresis states
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

% state scales from the measured response: displacements, tilts (over a
% unit span) and their rates at the highest spin speed
sx = std(x_meas); sy = std(y_meas); w = max(abs(Omega_fun(t(:))));
sc = [sx sy sx sy w*sx w*sy w*sx w*sy].^2;
P0 = diag(sc);
Q  = o.Q_scale * diag(sc);

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
    Phi = expm(dt*Fjac);                     % discretized linearization
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
if isfield(P, 'bw'), P.bw.on = false; end
end

function d = result_dir()
d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results');
if ~exist(d, 'dir'), mkdir(d); end
end
