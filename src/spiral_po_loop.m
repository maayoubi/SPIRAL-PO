function out = spiral_po_loop(D, eqn, varargin)
% SPIRAL_PO_LOOP  The iterative, gated SPIRAL-PO loop for one observable
% equation ('x' or 'y'): the gated network formulation (Sect. 2.2.9 of the
% paper; residual network, quasi-random projection, Gate 1 screening,
% Gate 2 cross-validated partial F-test, stopping guard). Used for the
% ablation study (Sect. 4.9, Table 5).
%
%   out = spiral_po_loop(D, 'x')                      % gated loop, valid statistics
%   out = spiral_po_loop(D, 'x', 'preset', 'pointwise')  % pointwise tests, stratified folds
%   out = spiral_po_loop(D, 'x', 'projection', 'direct')
%
% D is a record in simulate_rotor format.
%
% One iteration: fit the residual network r ~ f_NN(xi) on the training
% chunks -> stopping guard -> evaluate the network (or the measured
% residual) on a projection set -> regress it on the remaining candidates
% (Gate 1 screen, Bonferroni t-test) -> cluster aliased survivors
% (|corr| > 0.7) -> admit representatives one at a time by Gate 2 ->
% update the residual. Stops when the guard fires, Gate 1 or Gate 2
% admits nothing, or the library is exhausted.
%
% Options (value of the 'valid' preset first)
%   'preset'      'valid' | 'pointwise' (sets the five options below)
%   'projection'  'halton_tube': NN on quasi-random points in a tube of
%                 radius 'tube_sigma' (in units of each coordinate's std)
%                 around the trajectory | 'halton_box': NN on Halton
%                 points filling the bounding box of xi (pointwise preset) |
%                 'nn_traj': NN on the trajectory | 'direct': no NN, the
%                 measured residual itself
%   'cv_mode'     'blocked': contiguous time blocks with a gap of one SG
%                 window; decision on the pooled out-of-block error |
%                 'rolling': forward chaining | 'stratified':
%                 amplitude-stratified folds, majority vote (pointwise preset)
%   'whiteness'   stopping guard. 'predictability': stop when the network
%                 no longer predicts held-out chunks (one-sided t-test of
%                 the per-chunk log MSE ratio) | 'lagspaced': Ljung-Box on
%                 samples one SG window apart | 'raw': Ljung-Box on all
%                 samples (pointwise preset; invalid for differentiated data)
%   'fdof'        Gate-2 F-test denominator degrees of freedom:
%                 'effective' (N/win) | 'raw' (N - p, pointwise preset)
%   'rep_rule'    cluster representative: 'trajectory' (member most
%                 correlated with the residual on the trajectory) |
%                 'gate1' (largest screening t, pointwise preset)
%   'use_seed'    true | false (ablation: no physics seed)
%   'gating'      true | false (ablation: one NN fit, one projection,
%                 sequentially thresholded least squares, no tests)
%   'tube_sigma' 0.1, 'max_iter' 6, 'NH' 2e4 (projection points),
%   'nn_hidden' [12 12], 'alpha' 0.05, 'lb_lags' 20, 'n_folds' 7,
%   'maj_frac' 0.5, 'seed_rng' 1, 'stlsq_tol' 0.05, 'verbose' true
%
% Output struct
%   .active_names, .active_coeffs   admitted closure terms and their
%                 coefficients (fitted to the post-seed residual)
%   .terms        per admitted term: name, iter, gate1_t (screening
%                 diagnostic), gate2_frac, gate2_F, gate2_p, gate2_df,
%                 gate2_gain
%   .termination  reason the loop stopped
%   .lb           guard history per iteration: [Q, crit, p, df, NN R2, p_NN]
%   .closure_R2adj, .seed, .final_residual, .target, .opts, .eqn

p = inputParser;
p.addParameter('preset', 'valid');
p.addParameter('projection', '');
p.addParameter('tube_sigma', 0.1);
p.addParameter('cv_mode', '');
p.addParameter('whiteness', '');
p.addParameter('fdof', '');
p.addParameter('use_seed', true);
p.addParameter('gating', true);
p.addParameter('max_iter', 6);
p.addParameter('NH', 2e4);
p.addParameter('nn_hidden', [12 12]);
p.addParameter('alpha', 0.05);
p.addParameter('lb_lags', 20);
p.addParameter('n_folds', 7);
p.addParameter('maj_frac', 0.5);
p.addParameter('seed_rng', 1);
p.addParameter('stlsq_tol', 0.05);
p.addParameter('rep_rule', '');
p.addParameter('verbose', true);
p.parse(varargin{:});
o = p.Results;

switch lower(o.preset)
    case 'valid',     def = {'halton_tube','blocked','predictability','effective','trajectory'};
    case 'pointwise', def = {'halton_box','stratified','raw','raw','gate1'};
    otherwise, error('spiral_po_loop:preset','unknown preset %s', o.preset);
end
if isempty(o.projection), o.projection = def{1}; end
if isempty(o.cv_mode),    o.cv_mode    = def{2}; end
if isempty(o.whiteness),  o.whiteness  = def{3}; end
if isempty(o.fdof),       o.fdof       = def{4}; end
if isempty(o.rep_rule),   o.rep_rule   = def{5}; end

say = @(varargin) o.verbose && fprintf(varargin{:});

t = D.t(:); x = D.x(:); y = D.y(:);
xdot = D.xd(:); ydot = D.yd(:); Omega = D.Omega(:); Omegadot = D.Omegadot(:);
if strcmpi(eqn,'x')
    x_t = x; xdot_t = xdot; xddot_t = D.xdd(:); win = D.win_x;
elseif strcmpi(eqn,'y')
    x_t = y; xdot_t = ydot; xddot_t = D.ydd(:); win = D.win_y;
else
    error('spiral_po_loop:badEqn', 'eqn must be ''x'' or ''y''');
end
N = numel(t);
rand('seed', o.seed_rng); randn('seed', o.seed_rng);

%% --- folds and NN train/validation split ---
[fold_id, idx_tr, idx_val] = make_folds(t, x, y, o, win);
Kfold = max(fold_id);

Xi_full = [x, y, xdot, ydot, Omega];

%% --- physics seed (or none) ---
theta_full = cumtrapz(t, Omega);
seed_cols = [Omega.^2.*cos(theta_full), Omega.^2.*sin(theta_full), -xdot_t, -x_t];
seed_names = {'unb_cos','unb_sin','-self_dot','-self'};
if o.use_seed
    seed = fit_seed(t(idx_tr), x_t(idx_tr), xdot_t(idx_tr), xddot_t(idx_tr), Omega(idx_tr));
    xddot_seed_full = seed.predict(t, x_t, xdot_t, Omega, theta_full);
    say('[seed, %s-eq] R2adj = %.5f\n', upper(eqn), seed.R2adj);
else
    seed = struct('coeffs', zeros(4,1), 'R2adj', 0);
    xddot_seed_full = zeros(N,1);
end
target = xddot_t - xddot_seed_full;

lib = @(xx,yy,xxd,yyd,OO,OOd,th) library_eval(xx,yy,xxd,yyd,OO,OOd,th,eqn,o.use_seed);
[~, all_names] = lib(x(1),y(1),xdot(1),ydot(1),Omega(1),Omegadot(1),theta_full(1));
[Phi_traj_all, ~] = lib(x,y,xdot,ydot,Omega,Omegadot,theta_full);

active_names = {}; active_terms_full = zeros(N,0); active_coeffs = [];
terms = struct('name',{},'iter',{},'gate1_t',{},'gate2_frac',{},'gate2_F',{},'gate2_p',{},'gate2_df',{},'gate2_gain',{});
lbh = zeros(0,6);
termination = sprintf('max_iter (%d) reached', o.max_iter);
r = target;

%% ===== ablation: no statistical gating (one NN, one projection, STLSQ) =====
if ~o.gating
    net = nn_fit(Xi_full(idx_tr,:), r(idx_tr), Xi_full(idx_val,:), r(idx_val), 'hidden', o.nn_hidden, 'seed', 1);
    [Phi_P, rhat_P] = projection_set(o, r, net, x,y,xdot,ydot,Omega,Omegadot,theta_full, lib, Phi_traj_all);
    keep = stlsq(Phi_P, rhat_P, o.stlsq_tol);
    active_names = all_names(keep);
    active_terms_full = Phi_traj_all(:, keep);
    if any(keep), active_coeffs = ols_std(active_terms_full, target); end
    for j = find(keep(:)')
        terms(end+1) = struct('name',all_names{j},'iter',1,'gate1_t',NaN,'gate2_frac',NaN,'gate2_F',NaN,'gate2_p',NaN,'gate2_df',NaN,'gate2_gain',NaN); %#ok<AGROW>
    end
    termination = 'no gating (single STLSQ pass)';
    r = target - active_terms_full*active_coeffs;
    out = pack_output(eqn, seed, o, active_names, active_coeffs, terms, termination, lbh, idx_tr, idx_val, fold_id, r, target, seed_names, win, active_terms_full);
    return;
end

for n_iter = 1:o.max_iter
    say('\n=== [%s-eq] iteration %d (%s, %s CV) ===\n', upper(eqn), n_iter, o.projection, o.cv_mode);

    % residual network (used by the stopping guard and the projection)
    net = nn_fit(Xi_full(idx_tr,:), r(idx_tr), Xi_full(idx_val,:), r(idx_val), 'hidden', o.nn_hidden, 'seed', n_iter);
    [is_pred, pPR, R2nn] = predictability_test(net, Xi_full, r, idx_val);
    ow = o; if strcmpi(ow.whiteness,'predictability'), ow.whiteness = 'lagspaced'; end
    [is_white, QLB, QLBcrit, pLB, dfLB] = whiteness_test(r, ow, win);
    lbh(end+1,:) = [QLB, QLBcrit, pLB, dfLB, R2nn, pPR]; %#ok<AGROW>
    say('guard: held-out NN R2 = %.4f (p = %.3g); Ljung-Box (%s) Q=%.1f crit=%.1f p=%.3g\n', ...
        R2nn, pPR, ow.whiteness, QLB, QLBcrit, pLB);
    if strcmpi(o.whiteness,'predictability')
        if ~is_pred
            termination = 'guard: residual not predictable from xi (held-out)'; break;
        end
    elseif strcmpi(o.whiteness,'raw')
        [is_white, QLB, QLBcrit, pLB] = whiteness_test(r(idx_val), o, win);
        lbh(end,1:3) = [QLB, QLBcrit, pLB];
        if is_white, termination = 'whiteness guard (residual white)'; break; end
    elseif is_white
        termination = 'whiteness guard (residual white)'; break;
    end

    available = setdiff(all_names, active_names, 'stable');
    if isempty(available)
        termination = 'library exhausted'; break;
    end

    %% projection set: NN surface (or measured residual) on chosen points
    [Phi_P, rhat_P, names_P] = projection_set(o, r, net, x,y,xdot,ydot,Omega,Omegadot,theta_full, lib, Phi_traj_all);
    keepc = ismember(names_P, available);
    Phi_P = Phi_P(:, keepc); names_P = names_P(keepc);

    %% Gate 1: screening only (ranks candidates; t-values are diagnostics)
    [bH, seH] = ols_std(Phi_P, rhat_P);
    pH = numel(names_P);
    neff1 = size(Phi_P,1);
    if strcmpi(o.projection,'direct') && strcmpi(o.fdof,'effective'), neff1 = floor(neff1/win); end
    [pass1, tstat] = gate1_bonferroni(bH, seH, neff1, pH, o.alpha);
    S_names = names_P(pass1);
    if isempty(S_names)
        termination = 'Gate 1: no candidate passes screening'; break;
    end
    [~, ord] = sort(abs(tstat(pass1)), 'descend');
    S_names = S_names(ord); S_t = tstat(pass1); S_t = S_t(ord);

    %% cluster aliased survivors (|corr| > 0.7 on the trajectory)
    S_idx = cellfun(@(s) find(strcmp(all_names,s)), S_names);
    C = abs(corr_safe(Phi_traj_all(idx_tr, S_idx)));
    reps = {}; reps_t = [];
    used = false(numel(S_names),1);
    for ii = 1:numel(S_names)
        if used(ii), continue; end
        members = find(~used & C(ii,:)' > 0.7); members = union(ii, members);
        used(members) = true;
        pick = ii;
        if strcmpi(o.rep_rule,'trajectory') && numel(members) > 1
            % representative: member most correlated with the residual
            gain = zeros(numel(members),1);
            for mm = 1:numel(members)
                cc = Phi_traj_all(:, S_idx(members(mm)));
                Cm = corr_safe([cc, r]); gain(mm) = abs(Cm(1,2));
            end
            [~, imax] = max(gain); pick = members(imax);
        end
        if numel(members) > 1
            say('cluster {%s} -> representative %s\n', strjoin(S_names(members),', '), S_names{pick});
        end
        reps{end+1} = S_names{pick}; reps_t(end+1) = S_t(pick); %#ok<AGROW>
    end

    %% Gate 2: cross-validated partial F-test, one representative at a time
    admitted = {};
    for si = 1:numel(reps)
        cname = reps{si};
        ccol = Phi_traj_all(:, strcmp(all_names, cname));
        already = active_terms_full(idx_tr,:);
        if strcmpi(o.preset,'valid') && o.use_seed
            % screen against the seed regressors
            already = [seed_cols(idx_tr,:), already];
        end
        if ~isempty(already) && vif_against(already, ccol(idx_tr)) > 10
            say('Gate 2: skip %-16s (VIF > 10 against admitted terms)\n', cname);
            continue;
        end
        g2 = gate2_cv(active_terms_full, ccol, target, fold_id, o, win);
        say('Gate 2: %-16s %d/%d folds, pooled held-out gain %.2f%%, F=%.2f (df %d,%d), p=%.3g [%s]\n', cname, ...
            round(g2.frac_pass*Kfold), Kfold, 100*g2.heldout_gain, g2.F, 1, g2.df2, g2.p, ternary(g2.accept,'accept','reject'));
        if ~g2.accept, continue; end

        trial = [active_terms_full, ccol];
        b_full = ols_std(trial, target);
        R2b = 0; if ~isempty(active_terms_full), R2b = r2adj(active_terms_full, target); end
        if r2adj(trial, target) <= R2b, continue; end
        contrib = abs(b_full) .* sqrt(mean(trial.^2,1))' / max(sqrt(mean(target.^2)), eps);
        if any(contrib > 20)
            say('   plausibility safeguard: %s rejected (collinearity artifact)\n', cname);
            continue;
        end
        active_terms_full = trial; active_names{end+1} = cname; active_coeffs = b_full; %#ok<AGROW>
        admitted{end+1} = cname; %#ok<AGROW>
        terms(end+1) = struct('name',cname,'iter',n_iter,'gate1_t',reps_t(si), ...
            'gate2_frac',g2.frac_pass,'gate2_F',g2.F,'gate2_p',g2.p,'gate2_df',g2.df2,'gate2_gain',g2.heldout_gain); %#ok<AGROW>
        say('   --> ADMITTED %s\n', cname);
    end
    if isempty(admitted)
        termination = 'Gate 2: nothing admitted'; break;
    end
    r = target - active_terms_full*active_coeffs;
end
out = pack_output(eqn, seed, o, active_names, active_coeffs, terms, termination, lbh, idx_tr, idx_val, fold_id, r, target, seed_names, win, active_terms_full);

end


%% ======================================================================
function out = pack_output(eqn, seed, o, active_names, active_coeffs, terms, termination, lbh, idx_tr, idx_val, fold_id, r, target, seed_names, win, active_terms_full)
out.eqn = eqn; out.seed = seed; out.opts = o;
out.active_names = active_names; out.active_coeffs = active_coeffs;
out.terms = terms; out.termination = termination; out.lb = lbh;
out.idx_tr = idx_tr; out.idx_val = idx_val; out.fold_id = fold_id;
out.final_residual = r; out.target = target;
out.seed_names = seed_names; out.win = win;
out.closure_R2adj = 0;
if ~isempty(active_terms_full), out.closure_R2adj = r2adj(active_terms_full, target); end
if o.verbose
    fprintf('[%s-eq] terms: %s | closure R2adj = %.4f | stop: %s\n', upper(eqn), ...
        strjoin(active_names, ', '), out.closure_R2adj, termination);
end
end

function [Phi, names] = library_eval(x,y,xd,yd,Om,Omd,th,eqn,use_seed)
[Phi, names] = build_library(x,y,xd,yd,Om,Omd,eqn);
if ~use_seed
    if strcmpi(eqn,'x'), s = x; sd = xd; else, s = y; sd = yd; end
    Phi = [Phi, Om.^2.*cos(th), Om.^2.*sin(th), -sd, -s];
    names = [names, {'unb_cos','unb_sin','-self_dot','-self'}];
end
end

function [Phi_P, rhat_P, names] = projection_set(o, r, net, x,y,xd,yd,Om,Omd,th, lib, Phi_traj)
N = numel(x);
if strcmpi(o.projection, 'direct')
    Phi_P = Phi_traj; rhat_P = r;
    [~, names] = lib(x(1),y(1),xd(1),yd(1),Om(1),Omd(1),th(1));
    return;
end
Z = [x, y, xd, yd, Om, Omd, th];
switch lower(o.projection)
    case 'nn_traj'
        P = Z;
    case 'halton_box'
        H = halton_seq(o.NH, 6);
        lo = min(Z(:,1:6)); hi = max(Z(:,1:6));
        P = zeros(o.NH, 7);
        for j = 1:6, P(:,j) = lo(j) + H(:,j)*(hi(j)-lo(j)); end
        P(:,7) = 0;   % unbalance phase is not part of xi (only used without seed)
    case 'halton_tube'
        H = halton_seq(o.NH, 7);
        k = min(N, 1 + floor(H(:,1)*N));             % quasi-random trajectory points
        sd = std(Z(:,1:6));
        g = sqrt(2)*erfinv(2*min(max(H(:,2:7),1e-9),1-1e-9) - 1);   % quasi-normal jitter
        P = Z(k,:);
        P(:,1:6) = P(:,1:6) + o.tube_sigma * g .* sd;
    otherwise
        error('spiral_po_loop:projection', 'unknown projection %s', o.projection);
end
rhat_P = net.predict(P(:,1:5));
[Phi_P, names] = lib(P(:,1),P(:,2),P(:,3),P(:,4),P(:,5),P(:,6),P(:,7));
end

function [fold_id, idx_tr, idx_val] = make_folds(t, x, y, o, win)
N = numel(t); K = o.n_folds;
switch lower(o.cv_mode)
    case 'stratified'
        % amplitude-stratified blocked folds (pointwise preset)
        r_amp = sqrt(x.^2+y.^2); dt = t(2)-t(1);
        min_block_len = max(20, round(0.005/dt));
        block_len = max(min_block_len, floor(N/(8*3*K)));
        nb = ceil(N/block_len); bid = min(nb, floor((0:N-1)'/block_len)+1);
        bamp = accumarray(bid, r_amp, [nb 1], @mean);
        [~, ordb] = sort(bamp); ns = max(2, min(8, floor(nb/(3*K))));
        strata = ceil((1:nb)*ns/nb); sob = zeros(nb,1); sob(ordb) = strata;
        trb = [];
        for s = 1:ns
            bs = find(sob==s); bs = bs(randperm(numel(bs)));
            trb = [trb; bs(1:round(0.7*numel(bs)))]; %#ok<AGROW>
        end
        idx_tr = find(ismember(bid,trb)); idx_val = find(~ismember(bid,trb));
        fold_id = zeros(N,1); rc = 0;
        for s = randperm(ns)
            bs = find(sob==s); bs = bs(randperm(numel(bs)));
            for b = bs', fold_id(bid==b) = mod(rc,K)+1; rc = rc+1; end
        end
    case {'blocked','rolling'}
        % K contiguous blocks in time. 'blocked': each block is a
        % validation fold for a model trained on the other blocks, with a
        % gap of one SG window removed on both sides (fold_id = 0 marks
        % samples excluded from that fold's training set). 'rolling': fold k validates block k+1 using only
        % blocks 1..k for training.
        edges = round(linspace(0, N, K+1));
        fold_id = zeros(N,1);
        for k = 1:K, fold_id(edges(k)+1:edges(k+1)) = k; end
        % NN early-stopping split: alternate contiguous 0.2 s chunks 7:3
        chunk = max(4*win, round(0.1/(t(2)-t(1))));
        cid = floor((0:N-1)'/chunk);
        isval = mod(cid,10) >= 7;
        idx_tr = find(~isval); idx_val = find(isval);
    otherwise
        error('spiral_po_loop:cv', 'unknown cv_mode %s', o.cv_mode);
end
end

function g = gate2_cv(Phi0, cand, y, fold_id, o, win)
% 'stratified' (pointwise preset): majority vote of per-fold partial F-tests with
% validation rollback. 'blocked' / 'rolling': the candidate must reduce the
% pooled out-of-block error and pass the pooled effective-sample-size
% F-test.
K = max(fold_id); pass = zeros(K,1); F = zeros(K,1); dfs = zeros(K,1);
sse0 = 0; sse1 = 0;
for k = 1:K
    val = (fold_id == k);
    switch lower(o.cv_mode)
        case 'blocked'
            iv = find(val); gap = false(size(val));
            gap(max(1,iv(1)-win):iv(1)-1) = true;
            gap(iv(end)+1:min(numel(val),iv(end)+win)) = true;
            tr = ~val & ~gap;
        case 'rolling'
            if k == 1, F(k) = NaN; pass(k) = NaN; continue; end
            iv = find(val);
            tr = (fold_id < k); tr(max(1,iv(1)-win):iv(1)-1) = false;
        otherwise
            tr = ~val;
    end
    r = partialF(Phi0(tr,:), y(tr), Phi0(val,:), y(val), cand(tr), cand(val), o, win);
    pass(k) = r.accept; F(k) = r.F; dfs(k) = r.df2;
    sse0 = sse0 + r.sse_val0; sse1 = sse1 + r.sse_val1;
end
ok = ~isnan(F);
g.frac_pass = mean(pass(ok));
% pooled statistic on the full record (reported with the decision)
rf = partialF(Phi0, y, Phi0, y, cand, cand, o, win);
g.F = rf.F; g.df2 = rf.df2; g.p = 1 - fcdf_local(rf.F, 1, rf.df2);
g.heldout_gain = 1 - sse1/max(sse0, realmin);
if any(strcmpi(o.cv_mode, {'blocked','rolling'}))
    g.accept = (sse1 < sse0) && (g.p < o.alpha);
else
    g.accept = g.frac_pass >= o.maj_frac;
end
end

function r = partialF(P0tr, ytr, P0val, yval, Ctr, Cval, o, win)
if isempty(P0tr)
    RSS0 = sum((ytr-mean(ytr)).^2); yv0 = zeros(size(yval)); p0 = 0;
else
    b0 = ols_std(P0tr, ytr); RSS0 = sum((ytr-P0tr*b0).^2); yv0 = P0val*b0; p0 = size(P0tr,2);
end
b1 = ols_std([P0tr Ctr], ytr); RSS1 = sum((ytr-[P0tr Ctr]*b1).^2);
n = numel(ytr);
if strcmpi(o.fdof,'effective'), neff = floor(n/win); else, neff = n; end
df2 = max(neff - p0 - 1, 1);
% Partial F with an effective sample size: with MA(win-1) errors the
% classical statistic is inflated by roughly n/neff, so it is rescaled
% by neff/n and referred to F(1, neff - p0 - 1).
r.F = (RSS0-RSS1) / (RSS1/max(n-p0-1,1)) * (neff/n);
r.df2 = df2;
Fcrit = finv_local(1-o.alpha, 1, df2);
v0 = sqrt(mean((yval-yv0).^2)); v1 = sqrt(mean((yval-[P0val Cval]*b1).^2));
r.sse_val0 = numel(yval)*v0^2; r.sse_val1 = numel(yval)*v1^2;
r.accept = (r.F > Fcrit) && (v1 <= v0*1.005);
end

function [is_pred, pval, R2] = predictability_test(net, Xi, r, idx_val)
% Held-out predictability of the residual from xi. Validation samples are
% grouped into their contiguous chunks; the per-chunk log MSE ratio of the
% zero predictor to the network is tested with a one-sided t-test.
iv = idx_val(:); brk = [0; find(diff(iv) > 1); numel(iv)];
rhat = net.predict(Xi(iv,:)); rv = r(iv);
nc = numel(brk)-1; d = zeros(nc,1);
for c = 1:nc
    s = brk(c)+1:brk(c+1);
    % per-chunk log of MSE(zero) / MSE(network)
    d(c) = log(mean(rv(s).^2) / max(mean((rv(s)-rhat(s)).^2), realmin));
end
R2 = 1 - sum((rv-rhat).^2)/sum(rv.^2);
tt = mean(d)/(std(d)/sqrt(nc));
pval = 1 - tcdf_local(tt, nc-1);
is_pred = pval < 0.05;
end

function p = tcdf_local(x, df)
xt = df./(df+x.^2); p = 1 - 0.5*betainc(xt, df/2, 0.5); p(x<0) = 1 - p(x<0);
end

function [is_white, Q, crit, pval, m] = whiteness_test(r, o, win)
r = r(:);
if strcmpi(o.whiteness,'lagspaced'), r = r(1:win:end); end
r = r - mean(r); n = numel(r); m = min(o.lb_lags, floor(n/5));
c0 = sum(r.^2); rho = zeros(m,1);
for k = 1:m, rho(k) = sum(r(1:n-k).*r(1+k:n))/c0; end
Q = n*(n+2)*sum(rho.^2 ./ (n-(1:m)'));
pval = 1 - gammainc(Q/2, m/2);
crit = chi2inv_local(1-o.alpha, m);
is_white = Q < crit;
end

function keep = stlsq(Phi, y, tol)
[Phi_s, sc] = standardize_columns(Phi);
b = Phi_s \ y; keep = true(size(b));
for it = 1:10
    small = abs(b)*1 < tol*sqrt(mean(y.^2));
    keep = ~small;
    if ~any(keep), break; end
    b = zeros(size(b)); b(keep) = Phi_s(:,keep) \ y;
end
keep = keep(:)';
end

function [b, se] = ols_std(Phi, y)
if isempty(Phi), b = zeros(0,1); se = b; return; end
[Ps, sc] = standardize_columns(Phi);
[Q, R] = qr(Ps, 0); bs = R \ (Q'*y);
res = y - Ps*bs; n = size(Ps,1); p = size(Ps,2);
s2 = sum(res.^2)/max(n-p,1); Ri = inv(R);
b = bs./sc(:); se = sqrt(diag(s2*(Ri*Ri')))./sc(:);
end

function v = r2adj(Phi, y)
b = ols_std(Phi, y); n = numel(y); p = size(Phi,2);
v = 1 - (n-1)/max(n-p,1) * sum((y-Phi*b).^2)/sum((y-mean(y)).^2);
end

function C = corr_safe(A)
A = A - mean(A); s = sqrt(sum(A.^2)); s(s==0) = 1; A = A./s; C = A'*A;
end

function v = vif_against(P, c)
b = pinv(P)*c; res = c - P*b; SST = sum((c-mean(c)).^2);
if SST < 1e-300, v = 1; return; end
v = 1/max(1e-12, sum(res.^2)/SST);
end

function s = ternary(c,a,b), if c, s = a; else, s = b; end, end

function x = finv_local(prob, d1, d2)
lo = 0; hi = 1e4;
for it = 1:200, mid = (lo+hi)/2; if fcdf_local(mid,d1,d2) < prob, lo = mid; else, hi = mid; end, end
x = (lo+hi)/2;
end
function p = fcdf_local(x, d1, d2)
x = max(x, 0); p = betainc((d1*x)./(d1*x+d2), d1/2, d2/2);
end
function x = chi2inv_local(prob, k)
lo = 0; hi = max(50,5*k);
while gammainc(hi/2,k/2) < prob, hi = 2*hi; end
for it = 1:200, mid = (lo+hi)/2; if gammainc(mid/2,k/2) < prob, lo = mid; else, hi = mid; end, end
x = (lo+hi)/2;
end

%% ---------------- helpers (seed, Gate 1, Halton, scaling) ----------------
function seed = fit_seed(t, x, xdot, xddot, Omega)
% Physics seed fitted by OLS on the training split:
%   sdd = a1 Omega^2 cos(theta) + a2 Omega^2 sin(theta) - (c/m) sd - (k/m) s,
% theta = int Omega dt. Returns coeffs [a1; a2; c/m; k/m], se, R2adj and a
% predict(t, s, sd, Omega, theta) handle.

theta = cumtrapz(t, Omega);   % accumulated phase [rad]

Phi = [Omega.^2 .* cos(theta), Omega.^2 .* sin(theta), -xdot, -x];
y   = xddot;

[coeffs, se, ~, R2adj] = ols_fit(Phi, y);

seed.coeffs  = coeffs;
seed.se      = se;
seed.theta   = theta;
seed.R2adj   = R2adj;
seed.predict = @(tt, xx, xxdot, OOm, th) ...
    [OOm.^2.*cos(th), OOm.^2.*sin(th), -xxdot, -xx] * coeffs;

end


function [b, se, RSS, R2adj] = ols_fit(Phi, y)
% OLS via the economy QR decomposition.
[Q, R] = qr(Phi, 0);
b = R \ (Q' * y);
resid = y - Phi*b;
RSS = sum(resid.^2);
n = size(Phi,1); p = size(Phi,2);
sigma2 = RSS / max(n-p, 1);
Rinv = inv(R);
covb = sigma2 * (Rinv * Rinv');
se = sqrt(diag(covb));
SST = sum((y - mean(y)).^2);
R2adj = 1 - (n-1)/(max(n-p,1)) * RSS/SST;
end

function [pass, tstat, thresh] = gate1_bonferroni(b, se, n, p, alpha)
% Gate 1: two-sided Bonferroni t-screen |t_l| > t_{alpha/(2p), n-p}.
tstat = b ./ se;
df = max(n - p, 1);
thresh = tinv_local(1 - alpha/(2*p), df);
pass = abs(tstat) > thresh;

end


function x = tinv_local(prob, df)
% Inverse CDF of Student's t (bisection on the regularized incomplete beta).
if df <= 0, error('tinv_local: df must be positive'); end
lo = -1e4; hi = 1e4;
for it = 1:200
    mid = (lo+hi)/2;
    p = tcdf_local(mid, df);
    if p < prob
        lo = mid;
    else
        hi = mid;
    end
end
x = (lo+hi)/2;
end

function H = halton_seq(n, d)
% n points of the d-dimensional Halton sequence in [0,1)^d (first d prime
% bases, burn-in of 20 points).
primes_list = primes(200);
if d > numel(primes_list)
    error('halton_seq:dimTooLarge', 'Increase the prime list for d > %d', numel(primes_list));
end
bases = primes_list(1:d);

burn = 20;
H = zeros(n, d);
for j = 1:d
    H(:,j) = van_der_corput(burn + (1:n), bases(j));
end

end

function s = van_der_corput(idx, base)
idx = idx(:);
s = zeros(size(idx));
f = ones(size(idx)) / base;
i = idx;
while any(i > 0)
    digit = mod(i, base);
    s = s + digit .* f;
    f = f / base;
    i = floor(i / base);
end
end

function [Phi_s, scale] = standardize_columns(Phi)
% Unit-RMS column scaling (no centering).
scale = sqrt(mean(Phi.^2, 1));
scale(scale == 0) = 1;
Phi_s = Phi ./ scale;
end
