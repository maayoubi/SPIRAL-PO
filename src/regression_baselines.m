function R = regression_baselines(D, eqn, varargin)
% REGRESSION_BASELINES  One-shot regressions of an acceleration equation
% on [seed, full candidate library], used to compare inference methods
% (Table 3 and Sect. 4.9 of the paper), and the moving-block bootstrap
% of an identified model (Table 4).
%
%   R = regression_baselines(D, 'x')                       % all methods
%   R = regression_baselines(D, 'x', 'B', 20)              % fewer bootstrap draws
%   T = regression_baselines(D, 'x', 'bootstrap', names, 'B', 1000)
%
% Methods (each selects candidates with a two-sided Bonferroni test at
% family level alpha over the K candidates):
%   ols_naive   pointwise OLS on all samples, classical standard errors
%   ols_spaced  pointwise OLS on samples one SG window apart
%   ols_boot    pointwise OLS with moving-block bootstrap standard errors
%   weak        weak-form regression (weak_form.m)
%
% Output R: R.names (candidates after the seed-collinearity screen),
% R.collinear_with_seed, and for each method a struct with fields
% selected, t, coeff, se, crit.
%
% Bootstrap form: the seed and the listed closure terms are refitted
% jointly on the trajectory; blocks of rows (regressors and target) of
% length 'block_len' (default 4 SG windows) are resampled with
% replacement B times. Returns a struct array (one entry per column,
% seed first) with name, coeff, se_boot, ci_lo, ci_hi (95 % percentile),
% z, p_boot, se_ols, block_len, B, n_blocks. Columns that are linear
% combinations of earlier columns are not estimable and are returned as NaN.

if numel(varargin) >= 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'bootstrap')
    R = block_bootstrap(D, eqn, varargin{2}, varargin{3:end});
    return;
end
p = inputParser;
p.addParameter('alpha', 0.05);
p.addParameter('B', 500);
p.parse(varargin{:});
o = p.Results;

Wk = weak_form(D, eqn, 'alpha', o.alpha);
names = Wk.names; K = numel(names); ns = 4;
[P, ~, info] = build_library(D, eqn, 'joint', names);
y = info.target; win = info.win;
crit = sqrt(2)*erfinv(2*(1 - o.alpha/(2*K)) - 1);
R.names = names; R.names_all = Wk.names_all; R.collinear_with_seed = Wk.collinear_with_seed;

[b, se] = ols_se(P, y);
R.ols_naive = pack(b, se, names, ns, crit);

ii = 1:win:numel(y);
[b2, se2] = ols_se(P(ii,:), y(ii));
R.ols_spaced = pack(b2, se2, names, ns, crit);

T = block_bootstrap(D, eqn, names, 'B', o.B);
R.ols_boot = pack([T.coeff]', [T.se_boot]', names, ns, crit);

R.weak = struct('t', Wk.t, 'coeff', Wk.coeff, 'se', Wk.se, 'selected', {Wk.selected}, ...
                'crit', Wk.crit, 'n_eq', Wk.n_eq);
end

% -------------------------------------------------------------------------
function T = block_bootstrap(D, eqn, closure_names, varargin)
p = inputParser;
p.addParameter('B', 1000);
p.addParameter('block_len', []);
p.addParameter('seed', 1);
p.parse(varargin{:});
o = p.Results;

[P_all, names, info] = build_library(D, eqn, 'joint', closure_names);
y = info.target; win = info.win;
% drop columns collinear with earlier ones (VIF > 1000); reported as NaN
q_all = size(P_all,2);
sc_all = sqrt(mean(P_all.^2,1)); sc_all(sc_all == 0) = 1;
keep = false(1, q_all);
for j = 1:q_all
    pj = P_all(:,j) / sc_all(j);
    if any(keep)
        Pk = P_all(:,keep) ./ sc_all(keep);
        r = pj - Pk*(Pk \ pj);
        R2 = 1 - sum(r.^2)/sum(pj.^2);
    else
        R2 = 0;
    end
    keep(j) = R2 < 1 - 1e-3;
end
P = P_all(:, keep);
sc = sc_all(keep); Ps = P ./ sc;
N = size(P,1); q = size(P,2);
bs = Ps \ y; b = bs ./ sc(:);
res = y - Ps*bs;
s2 = sum(res.^2)/(N-q);
se_ols = sqrt(diag(s2*inv(Ps'*Ps))) ./ sc(:);

if isempty(o.block_len), o.block_len = 4*win; end
Lb = o.block_len; nb = ceil(N/Lb);
rand('seed', o.seed);
Bmat = zeros(o.B, q);
n_redrawn = 0;
for r = 1:o.B
    % redraw resamples with a rank-deficient design
    while true
        starts = 1 + floor(rand(nb,1)*(N-Lb+1));
        idx = bsxfun(@plus, starts, 0:Lb-1)'; idx = idx(:); idx = idx(1:N);
        Pr = Ps(idx,:);
        sv = svd(Pr, 0);
        if sv(end) > 1e-10*sv(1), break; end
        n_redrawn = n_redrawn + 1;
    end
    Bmat(r,:) = (Pr \ y(idx))' ./ sc(:)';
end
if n_redrawn > 0
    fprintf('   block bootstrap: %d rank-deficient resamples drawn again\n', n_redrawn);
end

T = struct('name',{},'coeff',{},'se_boot',{},'ci_lo',{},'ci_hi',{},'z',{},'p_boot',{},'se_ols',{},'block_len',{},'B',{},'n_blocks',{});
kidx = find(keep);
for jj = 1:q_all
    j = find(kidx == jj);
    if isempty(j)   % not estimable
        T(jj) = struct('name',names{jj},'coeff',NaN,'se_boot',NaN,'ci_lo',NaN,'ci_hi',NaN, ...
            'z',NaN,'p_boot',NaN,'se_ols',NaN,'block_len',Lb,'B',o.B,'n_blocks',nb);
        continue;
    end
    bj = sort(Bmat(:,j));
    lo = bj(max(1,round(0.025*o.B))); hi = bj(min(o.B,round(0.975*o.B)));
    se = std(Bmat(:,j));
    pb = min(1, 2*min(mean(Bmat(:,j) <= 0), mean(Bmat(:,j) >= 0)));
    T(jj) = struct('name',names{jj},'coeff',b(j),'se_boot',se,'ci_lo',lo,'ci_hi',hi, ...
        'z',b(j)/se,'p_boot',max(pb, 1/o.B),'se_ols',se_ols(j),'block_len',Lb,'B',o.B,'n_blocks',nb);
end
end

function out = pack(b, se, names, ns, crit)
tt = b(ns+1:end) ./ se(ns+1:end);
out.t = tt(:)'; out.coeff = b(ns+1:end)'; out.se = se(ns+1:end)';
out.selected = names(abs(tt(:)') > crit);
out.crit = crit;
end

function [b, se] = ols_se(P, y)
sc = sqrt(mean(P.^2,1)); sc(sc == 0) = 1; Ps = P ./ sc;
bs = Ps \ y; res = y - Ps*bs; n = size(P,1); q = size(P,2);
s2 = sum(res.^2)/(n-q);
b = bs./sc(:); se = sqrt(diag(s2*inv(Ps'*Ps)))./sc(:);
end
