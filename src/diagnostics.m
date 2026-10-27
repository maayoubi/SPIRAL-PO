function R = diagnostics(what, varargin)
% DIAGNOSTICS  Checks of an identified model (Sect. 2.4 and 4 of the paper).
%
%   E = diagnostics('heldout', eqn, closure_names, D_fit, D_test)
%       Out-of-sample check on an independent record. Seed and closure are
%       refitted jointly on D_fit and applied unchanged to D_test.
%       E.gain_pct      share of the held-out post-seed residual variance
%                       explained by the closure: 100(1 - mse_full/mse_seed)
%       E.gain_true_pct the same against the noise-free acceleration
%                       (synthetic records only, else NaN)
%       E.rmse_seed, E.rmse_full, E.coeffs, E.names (seed columns first)
%
%   K = diagnostics('predict', Ex, Ey, D, k_steps, stride)
%       k-step-ahead prediction of x, y with the identified observable
%       model (Ex, Ey from 'heldout'), started from the measured state
%       every 'stride' samples (default 50). Inputs: spin schedule only.
%       K.nrmse_x, K.nrmse_y (RMSE / channel std), K.rmse_x, K.rmse_y,
%       K.diverged (share of diverged starts), K.n_starts
%
%   C = diagnostics('conditioning', D, eqn)
%       Conditioning of the column-normalized signature matrix (seed + all
%       candidates): singular values, condition number kappa, mu_Phi =
%       lambda_min(N^-1 Phi' Phi), numerical rank; for the full library
%       (C.full) and for cluster representatives with |corr| <= 0.7 (C.reps).
%
%   P = diagnostics('predictability', D, eqn, closure_names, coeffs)
%       Step 7: is the residual of the identified model still predictable
%       from xi = [x y xd yd Omega]? A residual network (nn_fit.m) is
%       trained on alternating contiguous chunks and scored on the others;
%       P.R2_heldout and a one-sided p-value from the per-chunk log
%       MSE-ratio t-test. coeffs: joint coefficients (seed first); empty
%       -> OLS refit.
%
%   [cap, snr] = diagnostics('capacity', D, eqn)
%       Library capacity (Table 7): share of the noise-free post-seed
%       acceleration energy that the full library can represent on the
%       trajectory (R^2; epsilon_0 = sqrt(1 - R^2)), and the
%       signal-to-noise variance ratio of the post-seed residual.

switch lower(what)
    case 'heldout',        R = heldout(varargin{:});
    case 'predict',        R = predict_xy(varargin{:});
    case 'conditioning',   R = conditioning(varargin{:});
    case 'predictability', R = predictability(varargin{:});
    case 'capacity'
        [cap, snr] = capacity(varargin{:});
        R = struct('R2', cap, 'snr', snr, 'epsilon0', sqrt(max(0, 1-cap)));
    otherwise
        error('diagnostics:what', 'unknown check %s', what);
end
end

% ========================================================================
function E = heldout(eqn, closure_names, D_fit, D_test)
[Pf, names, If] = build_library(D_fit, eqn, 'joint', closure_names);
[Pt, ~, It]     = build_library(D_test, eqn, 'joint', closure_names);
yf = If.target; yt = It.target; ns = 4;
b_full = Pf \ yf;
b_seed = Pf(:,1:ns) \ yf;
e_full = yt - Pt*b_full;
e_seed = yt - Pt(:,1:ns)*b_seed;
E.rmse_seed = sqrt(mean(e_seed.^2));
E.rmse_full = sqrt(mean(e_full.^2));
E.gain_pct = 100*(1 - E.rmse_full^2/E.rmse_seed^2);
E.gain_true_pct = NaN;
if isfield(D_test, 'traj')
    if strcmpi(eqn,'x'), s0 = D_test.x_true; else, s0 = D_test.y_true; end
    [~, ydd0] = sg_derivative(s0, D_test.t(2)-D_test.t(1), It.win, 4);
    E.gain_true_pct = 100*(1 - mean((ydd0 - Pt*b_full).^2)/mean((ydd0 - Pt(:,1:ns)*b_seed).^2));
end
E.coeffs = b_full; E.names = names; E.eqn = eqn;
end

% ========================================================================
function R = predict_xy(Ex, Ey, D, k_steps, stride)
if nargin < 5, stride = 50; end
bx = Ex.coeffs; nx = Ex.names; by = Ey.coeffs; ny = Ey.names;
t = D.t; dt = t(2)-t(1); N = numel(t);
th = cumtrapz(t, D.Omega);
starts = 1:stride:(N-k_steps);
S = [D.x(starts)'; D.y(starts)'; D.xd(starts)'; D.yd(starts)'];
Om_f  = @(tt) interp1(t, D.Omega, tt, 'linear', 'extrap');
Omd_f = @(tt) interp1(t, D.Omegadot, tt, 'linear', 'extrap');
th_f  = @(tt) interp1(t, th, tt, 'linear', 'extrap');
tt0 = t(starts)';
f = @(tt, S) model_rhs(tt, S, bx, nx, by, ny, Om_f, Omd_f, th_f);
for k = 1:k_steps
    tk = tt0 + (k-1)*dt;
    k1 = f(tk, S); k2 = f(tk+dt/2, S+dt/2*k1); k3 = f(tk+dt/2, S+dt/2*k2); k4 = f(tk+dt, S+dt*k3);
    S = S + dt/6*(k1+2*k2+2*k3+k4);
end
idx = starts + k_steps;
ex = S(1,:)' - D.x(idx); ey = S(2,:)' - D.y(idx);
bad = ~isfinite(ex) | ~isfinite(ey) | abs(ex) > 1e3*std(D.x);
R.diverged = mean(bad);
R.rmse_x = sqrt(mean(ex(~bad).^2)); R.rmse_y = sqrt(mean(ey(~bad).^2));
R.nrmse_x = R.rmse_x/std(D.x); R.nrmse_y = R.rmse_y/std(D.y);
R.n_starts = numel(starts); R.k_steps = k_steps;
end

function dS = model_rhs(tt, S, bx, nx, by, ny, Om_f, Omd_f, th_f)
x = S(1,:)'; y = S(2,:)'; xd = S(3,:)'; yd = S(4,:)';
Om = Om_f(tt(:)); Omd = Omd_f(tt(:)); th = th_f(tt(:));
ax = model_eval(bx, nx, 'x', x, y, xd, yd, Om, Omd, th);
ay = model_eval(by, ny, 'y', x, y, xd, yd, Om, Omd, th);
dS = [xd'; yd'; ax'; ay'];
end

function a = model_eval(b, names, eqn, x, y, xd, yd, Om, Omd, th)
if strcmpi(eqn,'x'), s = x; sd = xd; else, s = y; sd = yd; end
a = [Om.^2.*cos(th), Om.^2.*sin(th), -sd, -s] * b(1:4);
if numel(names) > 4
    [L, ln] = build_library(x, y, xd, yd, Om, Omd, eqn);
    for k = 5:numel(names)
        a = a + b(k) * L(:, strcmp(ln, names{k}));
    end
end
end

% ========================================================================
function C = conditioning(D, eqn, varargin)
p = inputParser;
p.addParameter('with_seed', true);
p.addParameter('corr_thresh', 0.7);
p.parse(varargin{:});
o = p.Results;
if o.with_seed, [L, names] = build_library(D, eqn, 'full');
else,           [L, names] = build_library(D, eqn, 'library'); end
C.full = cond_stats(L); C.full.names = names;
% greedy clustering in library order (seed columns always kept)
A = L - mean(L); A = A ./ max(sqrt(sum(A.^2)), eps); Rc = abs(A'*A);
keep = true(1, numel(names)); ns = 4*o.with_seed;
for i = ns+1:numel(names)
    if ~keep(i), continue; end
    for j = i+1:numel(names)
        if keep(j) && Rc(i,j) > o.corr_thresh, keep(j) = false; end
    end
end
C.reps = cond_stats(L(:,keep)); C.reps.names = names(keep);
C.max_abs_corr = max(Rc(~eye(size(Rc))));
end

function s = cond_stats(P)
N = size(P,1);
cn = sqrt(mean(P.^2,1)); cn(cn == 0) = 1;
Pn = P ./ cn;
sv = svd(Pn/sqrt(N));
s.sv = sv(:)';
s.cond = sv(1)/max(sv(end), eps);
s.mu_Phi = sv(end)^2;
s.rank = sum(sv > max(size(Pn))*eps(sv(1))*sqrt(N));
end

% ========================================================================
function P = predictability(D, eqn, closure_names, coeffs)
[Pm, ~, info] = build_library(D, eqn, 'joint', closure_names);
y = info.target; win = info.win;
if nargin < 4 || isempty(coeffs), coeffs = Pm \ y; end
r = y - Pm*coeffs;
N = numel(r); chunk = max(4*win, round(0.1/(D.t(2)-D.t(1))));
cid = floor((0:N-1)'/chunk); isval = mod(cid,10) >= 7;
itr = find(~isval); iva = find(isval);
Xi = [D.x, D.y, D.xd, D.yd, D.Omega];
net = nn_fit(Xi(itr,:), r(itr), Xi(iva,:), r(iva), 'hidden', [12 12], 'seed', 1);
rh = net.predict(Xi(iva,:)); rv = r(iva);
brk = [0; find(diff(iva) > 1); numel(iva)]; nc = numel(brk)-1; d = zeros(nc,1);
for c = 1:nc
    sI = brk(c)+1:brk(c+1);
    d(c) = log(mean(rv(sI).^2)/max(mean((rv(sI)-rh(sI)).^2), realmin));
end
P.R2_heldout = 1 - sum((rv-rh).^2)/sum(rv.^2);
tt = mean(d)/(std(d)/sqrt(nc)); xt = (nc-1)/((nc-1)+tt^2);
pv = 0.5*betainc(xt, (nc-1)/2, 0.5); if tt < 0, pv = 1 - pv; end
P.p = pv; P.n_chunks = nc;
end

% ========================================================================
function [cap, snr] = capacity(D, eqn)
[S, ~, info] = build_library(D, eqn, 'seed');
L = build_library(D, eqn, 'library');
if strcmpi(eqn,'x'), s0 = D.x_true; else, s0 = D.y_true; end
[~, ydd0] = sg_derivative(s0, D.t(2)-D.t(1), info.win, 4);
r0 = ydd0 - S*(S\ydd0);                 % noise-free post-seed residual
P = [S L];
cap = 1 - sum((ydd0 - P*(P\ydd0)).^2)/sum(r0.^2);
snr = var(r0)/var(info.target - ydd0);
end
