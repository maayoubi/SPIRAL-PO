function W = weak_form(D, eqn, varargin)
% WEAK_FORM  The identification step of SPIRAL-PO (Sect. 2.2.4
% and 2.2.5 of the paper): weak-form (integral) regression of one
% acceleration equation on the physics seed and the candidate library,
% with a Bonferroni-corrected admission test.
%
%   W = weak_form(D, 'x')                    % select terms, x-equation
%   W = weak_form(D, 'y')                    % select terms, y-equation
%   W = weak_form(D, 'x', 'terms', sel)      % refit seed + given terms
%   W = weak_form(Ds, 's')                   % single-channel record
%
% Method. The equation  sdd = seed(theta) + sum_l theta_l phi_l  is
% multiplied by test functions psi_m(t) = (1 - tau^2)^4, tau = (t-t_m)/h,
% with disjoint supports of half-width h, and integrated. Derivatives of
% the measured signal are moved onto psi by integration by parts:
%     int psi sdd = int psi'' s,   int psi sd = -int psi' s,
%     int psi Omega yd = -int (psi' Omega + psi Omegadot) y,
%     int psi s^2 sd   = -int psi' s^3 / 3     (single channel),
% A candidate is admitted when |t| exceeds the
% two-sided Bonferroni critical value z_{1-alpha/(2K)} over the K
% candidates tested (K = 8: 2.73; K = 4: 2.50).
%
% Candidates that are numerically linear combinations of the seed columns
% (variance inflation > 1000 against the seed) are removed before the
% test and listed in W.collinear_with_seed.
%
% Inputs
%   D     record in simulate_rotor format. For eqn = 's' (single channel)
%         D needs t, x (the channel), xd, Omega, Omegadot only.
%   eqn   'x' | 'y' (bivariate library of build_library.m) or 's'
%         (single-channel library {s^3, s^2*sd, Omega^2*s, Omegadot*s})
% Options
%   'alpha'  family-wise level (0.05)
%   'h'      test-function half-width [s] (0.025)
%   'terms'  refit only the seed and these candidate names (cell array,
%            may be empty: seed only); no test
%
% Output W
%   names      candidate names tested (after the collinearity screen),
%              or the refitted names (seed first) with 'terms'
%   coeff, se, t   weak-form estimates for the candidates (selection) or
%              for all refitted columns (with 'terms')
%   b, se_all  all coefficients incl. the four seed columns
%   selected   admitted candidates; crit = critical |t| used
%   n_eq       number of test functions (independent equations)
%   G, Gy      the weak-form regression matrix and target
%   collinear_with_seed   candidates removed by the screen

p = inputParser;
p.addParameter('alpha', 0.05);
p.addParameter('h', 0.025);
p.addParameter('terms', NaN);
p.parse(varargin{:});
o = p.Results;
refit = iscell(o.terms) || ischar(o.terms);   % 'terms' given (possibly empty)
if ischar(o.terms), o.terms = {o.terms}; end

if strcmpi(eqn, 's')
    W = single_channel(D, o);
    return;
end

% ---------------- bivariate x- or y-equation ----------------
[S, ~, info] = build_library(D, eqn, 'seed');
[L, names] = build_library(D, eqn, 'library');
ns = 4;
if ~refit
    % collinearity screen against the seed (on the trajectory columns)
    colin = false(1, numel(names));
    for j = 1:numel(names)
        c = L(:,j); res = c - S*(S\c);
        colin(j) = sum(res.^2) < 1e-3*sum((c-mean(c)).^2) || all(c == 0);
    end
    W.collinear_with_seed = names(colin);
    W.names_all = names;
    cols = names(~colin);
else
    W.collinear_with_seed = {};
    cols = o.terms;
end
[G, Gy] = weak_system(D, eqn, S, L, names, cols, o.h);
[b, se] = ols_standardized(G, Gy);
W.G = G; W.Gy = Gy; W.n_eq = size(G,1);
W.b = b; W.se_all = se;
if ~refit
    K = numel(cols);
    crit = sqrt(2)*erfinv(2*(1 - o.alpha/(2*K)) - 1);
    tt = b(ns+1:end) ./ se(ns+1:end);
    W.names = cols;
    W.t = tt(:)'; W.coeff = b(ns+1:end)'; W.se = se(ns+1:end)';
    W.selected = cols(abs(W.t) > crit);
    W.crit = crit;
else
    W.names = [{'unb_cos','unb_sin','-self_dot','-self'}, cols];
    W.coeff = b'; W.se = se'; W.t = (b./se)';
end
W.win = info.win;
end

% -------------------------------------------------------------------------
function [G, Gy] = weak_system(D, eqn, S, L, lib_names, cols, h)
% Weak-form rows: one per test function, columns = seed + cols.
t = D.t; dt = t(2)-t(1); nh = round(h/dt);
if strcmpi(eqn,'x'), s = D.x; cross_vel = D.y; gyro = 'Omega*ydot';
else,                s = D.y; cross_vel = D.x; gyro = 'Omega*xdot'; end
[centers, psi, dpsi, ddpsi, w] = test_functions(numel(t), dt, h);
M = numel(centers); K = numel(cols);
Gy = zeros(M,1); G = zeros(M, 4+K);
for m = 1:M
    k = centers(m) + (-nh:nh)';
    Gy(m)  = sum(w .* ddpsi .* s(k));            % int psi*sdd = int psi''*s
    G(m,1) = sum(w .* psi .* S(k,1));
    G(m,2) = sum(w .* psi .* S(k,2));
    G(m,3) = sum(w .* dpsi .* s(k));             % int psi*(-sd) = int psi'*s
    G(m,4) = sum(w .* psi .* S(k,4));
    for j = 1:K
        if strcmp(cols{j}, gyro)                 % int psi*Om*vd = -int (psi*Om)'*v
            G(m,4+j) = -sum(w .* (dpsi.*D.Omega(k) + psi.*D.Omegadot(k)) .* cross_vel(k));
        else
            G(m,4+j) = sum(w .* psi .* L(k, strcmp(lib_names, cols{j})));
        end
    end
end
end

function W = single_channel(D, o)
% Single-channel record s(t) = D.x: seed + {s^3, s^2*sd, Omega^2*s, Omegadot*s}.
th = cumtrapz(D.t, D.Omega);
S = [D.Omega.^2.*cos(th), D.Omega.^2.*sin(th), -D.xd, -D.x];
L = [D.x.^3, D.x.^2.*D.xd, D.Omega.^2.*D.x, D.Omegadot.*D.x];
names = {'s^3','s^2*sdot','Omega^2*s','Omegadot*s'};
dt = D.t(2)-D.t(1); nh = round(o.h/dt);
[centers, psi, dpsi, ddpsi, w] = test_functions(numel(D.t), dt, o.h);
M = numel(centers); Gy = zeros(M,1); G = zeros(M,8);
for m = 1:M
    k = centers(m) + (-nh:nh)';
    Gy(m) = sum(w.*ddpsi.*D.x(k));
    G(m,1) = sum(w.*psi.*S(k,1)); G(m,2) = sum(w.*psi.*S(k,2));
    G(m,3) = sum(w.*dpsi.*D.x(k));
    G(m,4) = sum(w.*psi.*S(k,4));
    G(m,5) = sum(w.*psi.*L(k,1));
    G(m,6) = -sum(w.*dpsi.*D.x(k).^3)/3;         % s^2*sd = d(s^3/3)/dt
    G(m,7) = sum(w.*psi.*L(k,3));
    G(m,8) = sum(w.*psi.*L(k,4));
end
% collinearity screen against the seed, on the weak-form columns
keep = true(1,8);
for j = 5:8
    c = G(:,j); r = c - G(:,1:4)*(G(:,1:4)\c);
    keep(j) = sum(r.^2) > 1e-3*sum(c.^2);
end
cf = zeros(8,1); se = nan(8,1);
[cf(keep), se(keep)] = ols_standardized(G(:,keep), Gy);
W.names = names;
W.b = cf; W.se_all = se;
W.coeff = cf(5:8)'; W.se = se(5:8)'; W.t = (cf(5:8)./se(5:8))';
K = numel(names);
W.crit = sqrt(2)*erfinv(2*(1 - o.alpha/(2*K)) - 1);
W.selected = names(abs(W.t) > W.crit);
W.collinear_with_seed = names(~keep(5:8));
W.G = G; W.Gy = Gy; W.n_eq = M; W.S = S; W.L = L;
end

function [centers, psi, dpsi, ddpsi, w] = test_functions(N, dt, h)
% Test functions psi = (1-tau^2)^4 on disjoint windows of 2*nh+1 samples,
% their first two derivatives, and trapezoid quadrature weights.
nh = round(h/dt);
centers = (nh+1):(2*nh+1):(N-nh);
tau   = (-nh:nh)'*dt/h;
psi   = (1 - tau.^2).^4;
dpsi  = (-8*tau.*(1-tau.^2).^3)/h;
ddpsi = (-8*(1-tau.^2).^3 + 48*tau.^2.*(1-tau.^2).^2)/h^2;
w = dt*ones(2*nh+1,1); w([1 end]) = dt/2;
end

function [b, se] = ols_standardized(P, y)
% OLS on unit-RMS columns; coefficients returned in original units.
sc = sqrt(mean(P.^2, 1)); sc(sc == 0) = 1;
Ps = P ./ sc;
bs = Ps \ y; res = y - Ps*bs; n = size(P,1); q = size(P,2);
s2 = sum(res.^2)/(n-q);
b = bs./sc(:); se = sqrt(diag(s2*inv(Ps'*Ps)))./sc(:);
end
