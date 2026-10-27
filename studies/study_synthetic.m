function study_synthetic(part)
% STUDY_SYNTHETIC  All synthetic-benchmark results of the paper
% (Sect. 4.6-4.11, Tables 3-7). Writes CSV tables and plain-text logs to
% results/ and caches the simulated records in results/cache.
%
%   study_synthetic            % everything (several hours in Octave)
%   study_synthetic('A')       % steps 0-3                 (~1 h)
%   study_synthetic('B')       % steps 4-10                (~1 h)
%   study_synthetic('mc')      % Monte Carlo selection rates (Table 3; slow)
%   study_synthetic('coef')    % coefficient table (Table 4), residual predictability
%   study_synthetic('noise')   % weak-form selection versus noise level
%
% 'mc', 'coef' and 'noise' reuse the records cached by 'A' and 'B', so run
% those first. A and B can run in two sessions in parallel.
%
% Step  Content                                              Paper
%   0   model check: extended model at defaults == plain     --
%   1   nominal record: gated loops (pointwise / valid),     Sect. 4.6
%       held-out gain, bootstrap tables, conditioning (kappa, mu_Phi)
%   2   ablations and HAVOK baseline, 3 noise realizations   Table 5
%   3   knock-out controls with the gated loops              Sect. 4.8
%   4   structural mismatch with the gated loops             Sect. 4.10
%   5   cross-validation schemes                             Sect. 4.9
%   6   projection set / tube radius                         Sect. 4.9
%   7   one-shot regressions (OLS, spaced OLS, bootstrap,    Sect. 4.9
%       weak form) on the nominal record
%   8   library capacity epsilon_0                           Table 7
%   9   constant-speed rank check (simultaneous x, y)        Sect. 4.6
%  10   weak-form selection in every case                    Table 6
%  mc   selection rates over 20 noise realizations           Table 3
%  coef weak-form coefficients and bootstrap SEs             Table 4
%  noise weak-form selection at 0.5/2/5/10 % noise           Sect. 4.8
%
% Output files (results/): summary_<part>.txt, step1_nominal.csv,
% step1_boot_<pipeline>_<eq>.csv, step1_conditioning_<eq>.csv,
% step2_ablations.csv, step3_knockouts.csv, step4_mismatch.csv,
% step5_cv.csv, step6_projection.csv, step7_direct.csv,
% step8_capacity.csv, step10_weakform_cases.csv, extra_mc.txt/.csv,
% extra_coef.txt, extra_noise.txt/.csv

if nargin < 1, part = 'all'; end
RES = result_dir();
if strcmpi(part, 'all')
    study_synthetic('A'); study_synthetic('B');
    extra_studies('mc'); extra_studies('coef'); extra_studies('noise');
    return;
end
if any(strcmpi(part, {'mc','coef','noise'}))
    extra_studies(lower(part));
    return;
end
doA = strcmpi(part, 'A');
doB = strcmpi(part, 'B');
log_file = fullfile(RES, sprintf('summary_%s.txt', part));
fid = fopen(log_file, 'w');
say = @(varargin) say_both(fid, varargin{:});
reps = [31 41 51];          % noise realizations
test_seed = 77;             % independent record: different noise and phase

say('SPIRAL-PO revision studies, part %s, %s\n\n', part, datestr(now));

%% ---------------- step 0: model check ----------------
if doA
    % extended model at default settings versus the plain model
    cfg = spiral_config('nominal', 31); P = cfg.params;
    P0 = rmfield(P, {'aniso','gyro_scale','bw'});
    Om = @(tt) 2*pi*max(5, 60-55/8*tt); th = @(tt) 2*pi*(60*tt - 0.5*55/8*tt.^2);
    z = [1e-3; -2e-3; 3e-3; 1e-3; 0.1; -0.2; 0.05; 0.3]; dmax = 0;
    for tt = [0 1.3 4.7]
        dmax = max(dmax, max(abs(rotor_rhs(tt,z,P,Om,th) - rotor_rhs(tt,z,P0,Om,th))));
    end
    say('[0] max |rhs(extended defaults) - rhs(plain model)| = %.3g\n\n', dmax);
end

%% ---------------- step 1: nominal benchmark ----------------
if doA
    D  = get_data('nominal', 31);
    D2 = get_test('nominal');
    say('[1] Nominal benchmark (Table 1 configuration): %s\n', describe(D));
    rows = {};
    for pre = {'pointwise','valid'}
        for e = {'x','y'}
            o = spiral_po_loop(D, e{1}, 'preset', pre{1}, 'verbose', false);
            E = diagnostics('heldout', o.eqn, o.active_names, D, D2);
            rows(end+1,:) = {pre{1}, e{1}, strjoin(o.active_names,' + '), o.closure_R2adj, ...
                E.gain_pct, E.gain_true_pct, o.termination, size(o.lb,1)}; %#ok<AGROW>
            say('    %-7s %s-eq: %-60s | closure R2adj %.3f | held-out gain %6.2f%% (true closure %6.2f%%) | stop: %s\n', ...
                pre{1}, e{1}, strjoin(o.active_names,', '), o.closure_R2adj, E.gain_pct, E.gain_true_pct, o.termination);
            % iteration log (guard statistics)
            for k = 1:size(o.lb,1)
                say('        iter %d: Ljung-Box Q=%.1f (crit %.1f, p=%.3g, m=%d); held-out NN R2=%.3f (p=%.3g)\n', k, o.lb(k,1:6));
            end
            % bootstrap coefficient table for the admitted set
            T = regression_baselines(D, e{1}, 'bootstrap', o.active_names, 'B', 1000);
            write_boot(fullfile(RES, sprintf('step1_boot_%s_%s.csv', pre{1}, e{1})), T, o);
            for j = 1:numel(T)
                say('        %-16s %+.4e  SE_boot %.2e  SE_ols %.2e  95%% CI [%+.3e, %+.3e]  p_boot %.3g\n', ...
                    T(j).name, T(j).coeff, T(j).se_boot, T(j).se_ols, T(j).ci_lo, T(j).ci_hi, T(j).p_boot);
            end
        end
    end
    write_cell(fullfile(RES,'step1_nominal.csv'), ...
        {'pipeline','eqn','terms','closure_R2adj','heldout_gain_pct','true_closure_gain_pct','termination','iterations'}, rows);
    % conditioning of the signature matrix
    for e = {'x','y'}
        C = diagnostics('conditioning', D, e{1});
        say('    conditioning %s-eq: full library  kappa=%.3g mu_Phi=%.3g rank=%d/%d | representatives kappa=%.3g mu_Phi=%.3g (%s)\n', ...
            e{1}, C.full.cond, C.full.mu_Phi, C.full.rank, numel(C.full.names), C.reps.cond, C.reps.mu_Phi, strjoin(C.reps.names, ', '));
        write_cond(fullfile(RES, sprintf('step1_conditioning_%s.csv', e{1})), C);
    end
    say('\n');
end

%% ---------------- step 2: ablations ----------------
if doA
    say('[2] Ablations (noise realizations %s)\n', mat2str(reps));
    cfgs = { 'full (valid statistics)', {};
             'gated network (pointwise)', {'preset','pointwise'};
             'A2 no NN (direct)',     {'projection','direct'};
             'A3 no gating (STLSQ)',  {'gating',false};
             'A4 NN on trajectory',   {'projection','nn_traj'};
             'A4b NN on Halton box',  {'projection','halton_box'};
             'A5 no physics seed',    {'use_seed',false} };
    rows = {};
    D2 = get_test('nominal');
    for r = reps
        D = get_data('nominal', r);
        % A1: seed only
        Ex = diagnostics('heldout', 'x', {}, D, D2);
        Ey = diagnostics('heldout', 'y', {}, D, D2);
        K20 = diagnostics('predict', Ex, Ey, D2, 20, 50);
        K100 = diagnostics('predict', Ex, Ey, D2, 100, 50);
        rows(end+1,:) = {'A1 seed only', r, '', '', 0, 0, NaN, NaN, K20.nrmse_x, K20.nrmse_y, K100.nrmse_x, K100.nrmse_y, 0,0,0, '-','-'}; %#ok<AGROW>
        for c = 1:size(cfgs,1)
            ox = spiral_po_loop(D, 'x', cfgs{c,2}{:}, 'verbose', false);
            oy = spiral_po_loop(D, 'y', cfgs{c,2}{:}, 'verbose', false);
            Ex = diagnostics('heldout', ox.eqn, ox.active_names, D, D2); Ey = diagnostics('heldout', oy.eqn, oy.active_names, D, D2);
            K20 = diagnostics('predict', Ex, Ey, D2, 20, 50);
            K100 = diagnostics('predict', Ex, Ey, D2, 100, 50);
            [g, du, cr] = mechanisms(ox, oy);
            rows(end+1,:) = {cfgs{c,1}, r, strjoin(ox.active_names,' + '), strjoin(oy.active_names,' + '), ...
                Ex.gain_pct, Ey.gain_pct, Ex.gain_true_pct, Ey.gain_true_pct, ...
                K20.nrmse_x, K20.nrmse_y, K100.nrmse_x, K100.nrmse_y, g, du, cr, ox.termination, oy.termination}; %#ok<AGROW>
            say('    rep %d %-22s gain x/y %6.2f/%6.2f%% | k=20 NRMSE %.3f/%.3f | gyro %d duff %d cross %d | x: %s | y: %s\n', ...
                r, cfgs{c,1}, Ex.gain_pct, Ey.gain_pct, K20.nrmse_x, K20.nrmse_y, g, du, cr, ...
                strjoin(ox.active_names,', '), strjoin(oy.active_names,', '));
        end
        H20 = havok_baseline(D, D2, 20); H100 = havok_baseline(D, D2, 100);
        rows(end+1,:) = {'A6 HAVOK (forced)', r, sprintf('r=%d, q=%d', H20.r, H20.q), '', NaN, NaN, NaN, NaN, ...
            H20.nrmse_forced_x, H20.nrmse_forced_y, H100.nrmse_forced_x, H100.nrmse_forced_y, NaN,NaN,NaN,'-','-'}; %#ok<AGROW>
        rows(end+1,:) = {'A6 HAVOK (frozen forcing)', r, sprintf('r=%d, q=%d', H20.r, H20.q), '', NaN, NaN, NaN, NaN, ...
            H20.nrmse_frozen_x, H20.nrmse_frozen_y, H100.nrmse_frozen_x, H100.nrmse_frozen_y, NaN,NaN,NaN,'-','-'}; %#ok<AGROW>
        say('    rep %d HAVOK r=%d: k=20 NRMSE forced %.3f/%.3f, frozen %.3f/%.3f; k=100 forced %.3f/%.3f, frozen %.3f/%.3f\n', ...
            r, H20.r, H20.nrmse_forced_x, H20.nrmse_forced_y, H20.nrmse_frozen_x, H20.nrmse_frozen_y, ...
            H100.nrmse_forced_x, H100.nrmse_forced_y, H100.nrmse_frozen_x, H100.nrmse_frozen_y);
    end
    write_cell(fullfile(RES,'step2_ablations.csv'), {'config','noise_seed','x_terms','y_terms', ...
        'gain_x_pct','gain_y_pct','true_gain_x_pct','true_gain_y_pct','nrmse_k20_x','nrmse_k20_y', ...
        'nrmse_k100_x','nrmse_k100_y','gyro_both','duffing_both','cross_both','stop_x','stop_y'}, rows);
    say('\n');
end

%% ---------------- step 3: knock-out controls ----------------
if doA
    say('[3] Knock-out controls (a mechanism absent from the data must not be admitted)\n');
    rows = {};
    for cs = {'no_duffing','no_gyro'}
        D2 = get_test(cs{1});
        for r = reps
            D = get_data(cs{1}, r);
            for pre = {'pointwise','valid'}
                ox = spiral_po_loop(D, 'x', 'preset', pre{1}, 'verbose', false);
                oy = spiral_po_loop(D, 'y', 'preset', pre{1}, 'verbose', false);
                [g, du] = mechanisms(ox, oy, 'any');
                Ex = diagnostics('heldout', ox.eqn, ox.active_names, D, D2); Ey = diagnostics('heldout', oy.eqn, oy.active_names, D, D2);
                rows(end+1,:) = {cs{1}, r, pre{1}, strjoin(ox.active_names,' + '), strjoin(oy.active_names,' + '), g, du, Ex.gain_pct, Ey.gain_pct}; %#ok<AGROW>
                say('    %-10s rep %d %-7s gyro admitted %d, cubic admitted %d | gain %6.2f/%6.2f%% | x: %s | y: %s\n', ...
                    cs{1}, r, pre{1}, g, du, Ex.gain_pct, Ey.gain_pct, strjoin(ox.active_names,', '), strjoin(oy.active_names,', '));
            end
        end
    end
    write_cell(fullfile(RES,'step3_knockouts.csv'), {'case','noise_seed','pipeline','x_terms','y_terms', ...
        'gyro_in_any_eq','cubic_in_any_eq','gain_x_pct','gain_y_pct'}, rows);
    say('\n');
end

%% ---------------- step 4: structural mismatch ----------------
if doB
    say('[4] Structural mismatch (data-generating model differs from the assumed structure)\n');
    rows = {};
    for cs = {'aniso','hysteresis','params','runup'}
        D = get_data(cs{1}, 31); D2 = get_test(cs{1});
        for pre = {'pointwise','valid'}
            ox = spiral_po_loop(D, 'x', 'preset', pre{1}, 'verbose', false);
            oy = spiral_po_loop(D, 'y', 'preset', pre{1}, 'verbose', false);
            Ex = diagnostics('heldout', ox.eqn, ox.active_names, D, D2); Ey = diagnostics('heldout', oy.eqn, oy.active_names, D, D2);
            K20 = diagnostics('predict', Ex, Ey, D2, 20, 50);
            rows(end+1,:) = {cs{1}, pre{1}, strjoin(ox.active_names,' + '), strjoin(oy.active_names,' + '), ...
                Ex.gain_pct, Ey.gain_pct, Ex.gain_true_pct, Ey.gain_true_pct, K20.nrmse_x, K20.nrmse_y, ox.termination, oy.termination}; %#ok<AGROW>
            say('    %-10s %-7s gain %6.2f/%6.2f%% (true %6.2f/%6.2f%%) k20 %.3f/%.3f | x: %s [%s] | y: %s [%s]\n', cs{1}, pre{1}, ...
                Ex.gain_pct, Ey.gain_pct, Ex.gain_true_pct, Ey.gain_true_pct, K20.nrmse_x, K20.nrmse_y, ...
                strjoin(ox.active_names,', '), ox.termination, strjoin(oy.active_names,', '), oy.termination);
        end
    end
    write_cell(fullfile(RES,'step4_mismatch.csv'), {'case','pipeline','x_terms','y_terms','gain_x_pct','gain_y_pct', ...
        'true_gain_x_pct','true_gain_y_pct','nrmse_k20_x','nrmse_k20_y','stop_x','stop_y'}, rows);
    say('\n');
end

%% ---------------- step 5: CV schemes ----------------
if doB
    say('[5] Cross-validation scheme (valid-statistics loop otherwise)\n');
    D = get_data('nominal', 31); D2 = get_test('nominal'); rows = {};
    for cv = {'stratified','blocked','rolling'}
        ox = spiral_po_loop(D, 'x', 'cv_mode', cv{1}, 'verbose', false);
        oy = spiral_po_loop(D, 'y', 'cv_mode', cv{1}, 'verbose', false);
        Ex = diagnostics('heldout', ox.eqn, ox.active_names, D, D2); Ey = diagnostics('heldout', oy.eqn, oy.active_names, D, D2);
        rows(end+1,:) = {cv{1}, strjoin(ox.active_names,' + '), strjoin(oy.active_names,' + '), Ex.gain_pct, Ey.gain_pct}; %#ok<AGROW>
        say('    %-10s gain %6.2f/%6.2f%% | x: %s | y: %s\n', cv{1}, Ex.gain_pct, Ey.gain_pct, strjoin(ox.active_names,', '), strjoin(oy.active_names,', '));
    end
    % pointwise statistics (raw dof) with each CV scheme
    for cv = {'stratified','blocked'}
        ox = spiral_po_loop(D, 'x', 'preset','pointwise', 'cv_mode', cv{1}, 'verbose', false);
        oy = spiral_po_loop(D, 'y', 'preset','pointwise', 'cv_mode', cv{1}, 'verbose', false);
        Ex = diagnostics('heldout', ox.eqn, ox.active_names, D, D2); Ey = diagnostics('heldout', oy.eqn, oy.active_names, D, D2);
        rows(end+1,:) = {['pointwise stats + ' cv{1}], strjoin(ox.active_names,' + '), strjoin(oy.active_names,' + '), Ex.gain_pct, Ey.gain_pct}; %#ok<AGROW>
        say('    pointwise stats + %-10s gain %6.2f/%6.2f%% | x: %s | y: %s\n', cv{1}, Ex.gain_pct, Ey.gain_pct, strjoin(ox.active_names,', '), strjoin(oy.active_names,', '));
    end
    write_cell(fullfile(RES,'step5_cv.csv'), {'cv_scheme','x_terms','y_terms','gain_x_pct','gain_y_pct'}, rows);
    say('\n');
end

%% ---------------- step 6: projection / tube radius ----------------
if doB
    say('[6] Projection set (valid-statistics loop otherwise)\n');
    D = get_data('nominal', 31); D2 = get_test('nominal'); rows = {};
    opts = { 'direct (no NN)', {'projection','direct'};
             'NN on trajectory', {'projection','nn_traj'};
             'tube sigma=0.05', {'tube_sigma',0.05};
             'tube sigma=0.1',  {'tube_sigma',0.1};
             'tube sigma=0.3',  {'tube_sigma',0.3};
             'Halton box',      {'projection','halton_box'} };
    for c = 1:size(opts,1)
        ox = spiral_po_loop(D, 'x', opts{c,2}{:}, 'verbose', false);
        oy = spiral_po_loop(D, 'y', opts{c,2}{:}, 'verbose', false);
        Ex = diagnostics('heldout', ox.eqn, ox.active_names, D, D2); Ey = diagnostics('heldout', oy.eqn, oy.active_names, D, D2);
        rows(end+1,:) = {opts{c,1}, strjoin(ox.active_names,' + '), strjoin(oy.active_names,' + '), Ex.gain_pct, Ey.gain_pct}; %#ok<AGROW>
        say('    %-18s gain %6.2f/%6.2f%% | x: %s | y: %s\n', opts{c,1}, Ex.gain_pct, Ey.gain_pct, strjoin(ox.active_names,', '), strjoin(oy.active_names,', '));
    end
    write_cell(fullfile(RES,'step6_projection.csv'), {'projection','x_terms','y_terms','gain_x_pct','gain_y_pct'}, rows);
    say('\n');
end

%% ---------------- step 7: one-shot direct regressions ----------------
if doB
    say('[7] One-shot direct regressions on the full library (Bonferroni, alpha=0.05)\n');
    D = get_data('nominal', 31); D2 = get_test('nominal'); rows = {};
    for e = {'x','y'}
        R = regression_baselines(D, e{1});
        for m = {'ols_naive','ols_spaced','ols_boot','weak'}
            s = R.(m{1});
            E = diagnostics('heldout', e{1}, s.selected, D, D2);
            rows(end+1,:) = {e{1}, m{1}, strjoin(s.selected,' + '), mat2str(round(s.t*100)/100), E.gain_pct, E.gain_true_pct}; %#ok<AGROW>
            say('    %s-eq %-10s selected: %-55s held-out gain %6.2f%% (true %6.2f%%) |t| = %s\n', e{1}, m{1}, ...
                strjoin(s.selected,', '), E.gain_pct, E.gain_true_pct, mat2str(round(abs(s.t)*10)/10));
        end
    end
    say('    (library order: %s)\n', strjoin(R.names, ', '));
    write_cell(fullfile(RES,'step7_direct.csv'), {'eqn','method','selected','t_values','heldout_gain_pct','true_closure_gain_pct'}, rows);
    say('\n');
end

%% ---------------- step 8: closure capacity ----------------
if doB
    say('[8] Closure capacity: share of the noise-free post-seed acceleration the full library can represent (fit on the trajectory)\n');
    rows = {};
    for cs = {'nominal','no_duffing','no_gyro','aniso','hysteresis','params','runup'}
        D = get_data(cs{1}, 31);
        for e = {'x','y'}
            Cp = diagnostics('capacity', D, e{1}); cap = Cp.R2; nn = Cp.snr;
            rows(end+1,:) = {cs{1}, e{1}, cap, nn}; %#ok<AGROW>
            say('    %-10s %s-eq: library R2 on true closure = %.3f  (epsilon_0 ~ %.2f); signal/noise variance of post-seed residual = %.2f\n', ...
                cs{1}, e{1}, cap, sqrt(max(0,1-cap)), nn);
        end
    end
    write_cell(fullfile(RES,'step8_capacity.csv'), {'case','eqn','library_R2_true_closure','signal_to_noise_var'}, rows);
    say('\n');
end

%% ---------------- step 9: constant speed ----------------
if doB
    say('[9] Constant-speed (60 Hz) synthetic record, simultaneous x and y, 30 s, last 5 s (transient decayed)\n');
    cfg = spiral_config('nominal', 31); cfg.f_start = 60; cfg.f_end = 60; cfg.Tend = 30;
    D = simulate_rotor(cfg); keep = D.t >= 25;
    th = cumtrapz(D.t, D.Omega);
    for e = {'x','y'}
        if strcmp(e{1},'x'), s = D.x; sd = D.xd; else, s = D.y; sd = D.yd; end
        S = [D.Omega.^2.*cos(th), D.Omega.^2.*sin(th), -sd, -s];
        [L, nm] = build_library(D.x, D.y, D.xd, D.yd, D.Omega, D.Omegadot, e{1});
        % Omega^2*x is Omega0^2 times the seed column x and Omegadot*x is zero
        % at constant speed: both are removed before the rank check
        sel = ~ismember(nm, {'Omega^2*x','Omega^2*y','Omegadot*x','Omegadot*y'});
        P = [S L(:,sel)]; P = P(keep,:); P = P ./ sqrt(mean(P.^2,1));
        sv = svd(P/sqrt(size(P,1)));
        say('    %s-eq, seed + %s: singular values %s, kappa %.3g\n', e{1}, strjoin(nm(sel),', '), mat2str(sv',3), sv(1)/sv(end));
    end
    say('    (four singular values of order one and the rest at the noise level: the synchronous response spans ~4 directions)\n\n');
end

%% ---------------- step 10: weak-form selection across cases ----------------
if doB
    say('[10] Weak-form one-shot selection (Bonferroni, alpha=0.05) across all cases\n');
    rows = {};
    cases = {'nominal','no_duffing','no_gyro','aniso','hysteresis','params','runup'};
    for c = 1:numel(cases)
        seeds = 31; if any(strcmp(cases{c}, {'nominal','no_duffing','no_gyro'})), seeds = reps; end
        for r = seeds
            D = get_data(cases{c}, r);
            Rx = regression_baselines(D, 'x', 'B', 50); Ry = regression_baselines(D, 'y', 'B', 50);
            rows(end+1,:) = {cases{c}, r, strjoin(Rx.weak.selected,' + '), strjoin(Ry.weak.selected,' + '), strjoin([Rx.collinear_with_seed, Ry.collinear_with_seed],' + ')}; %#ok<AGROW>
            say('    %-10s rep %d  x: %-45s y: %s\n', cases{c}, r, strjoin(Rx.weak.selected,', '), strjoin(Ry.weak.selected,', '));
        end
    end
    write_cell(fullfile(RES,'step10_weakform_cases.csv'), {'case','noise_seed','x_terms','y_terms','removed_collinear_with_seed'}, rows);
    say('\n');
end

fclose(fid);
fprintf('\nWrote %s and CSV tables in %s\n', log_file, RES);
end


%% ======================================================================
function extra_studies(part)
% Monte Carlo ('mc'), coefficient table ('coef') and noise sweep ('noise').
% Uses the records cached by parts A/B.
RES = result_dir();
fid = fopen(fullfile(RES, sprintf('extra_%s.txt', part)), 'w');
say = @(varargin) say_both(fid, varargin{:});
cases = {'nominal','no_duffing','no_gyro'};
mech = {'Omega*ydot','(x^2+y^2)*x','y'; 'Omega*xdot','(x^2+y^2)*y','x'};
switch part
case 'mc'
    seeds = 101:120;
    say('Selection rates over %d noise realizations (2%% noise), Bonferroni alpha = 0.05\n', numel(seeds));
    say('%-11s %-11s %-4s | %-10s %-10s %-10s | %s\n','case','method','eqn','gyro','cubic*','cross','mean # selected terms');
    rows = {};
    for c = 1:numel(cases)
        S = load(fullfile(RES,'cache',sprintf('%s_31.mat', cases{c}))); D0 = S.D;
        for e = 1:2
            eq = 'xy'; eq = eq(e);
            cnt = struct('weak',zeros(1,4),'ols_naive',zeros(1,4),'ols_spaced',zeros(1,4));
            for s = seeds
                D = simulate_rotor(D0, 'renoise', s, 0.02);
                R = regression_baselines(D, eq, 'B', 20);
                for m = {'weak','ols_naive','ols_spaced'}
                    sel = R.(m{1}).selected;
                    cnt.(m{1}) = cnt.(m{1}) + [any(strcmp(sel, mech{e,1})), any(strcmp(sel, mech{e,2})), any(strcmp(sel, mech{e,3})), numel(sel)];
                end
            end
            for m = {'weak','ols_naive','ols_spaced'}
                v = cnt.(m{1})/numel(seeds);
                say('%-11s %-11s %-4s | %-10.2f %-10.2f %-10.2f | %.2f\n', cases{c}, m{1}, eq, v);
                rows(end+1,:) = {cases{c}, m{1}, eq, v(1), v(2), v(3), v(4), numel(seeds)}; %#ok<AGROW>
            end
        end
    end
    % gated network formulation, fewer realizations
    seeds2 = 101:110;
    for c = 1:numel(cases)
        S = load(fullfile(RES,'cache',sprintf('%s_31.mat', cases{c}))); D0 = S.D;
        for e = 1:2
            eq = 'xy'; eq = eq(e); v = zeros(1,4);
            for s = seeds2
                D = simulate_rotor(D0, 'renoise', s, 0.02);
                o = spiral_po_loop(D, eq, 'preset', 'pointwise', 'verbose', false);
                sel = o.active_names;
                v = v + [any(strcmp(sel, mech{e,1})), any(strcmp(sel, mech{e,2})), any(strcmp(sel, mech{e,3})), numel(sel)];
            end
            v = v/numel(seeds2);
            say('%-11s %-11s %-4s | %-10.2f %-10.2f %-10.2f | %.2f\n', cases{c}, 'gated', eq, v);
            rows(end+1,:) = {cases{c}, 'gated', eq, v(1), v(2), v(3), v(4), numel(seeds2)}; %#ok<AGROW>
        end
    end
    say('(* cubic = the isotropic term (x^2+y^2)x / (x^2+y^2)y)\n');
    write_cell(fullfile(RES,'extra_mc.csv'), {'case','method','eqn','rate_gyro','rate_cubic','rate_cross','mean_n_selected','n_realizations'}, rows);
case 'coef'
    S = load(fullfile(RES,'cache','nominal_31.mat')); D = S.D;
    say('Weak-form model, nominal benchmark (noise seed 31)\n');
    for e = {'x','y'}
        R = regression_baselines(D, e{1}, 'B', 1000);
        sel = R.weak.selected;
        % refit seed + selected terms in weak form, report coefficients
        Rw = weak_form(D, e{1}, 'terms', sel);
        T = regression_baselines(D, e{1}, 'bootstrap', sel, 'B', 1000);
        say('  %s-equation: selected %s\n', e{1}, strjoin(sel, ', '));
        for j = 1:numel(Rw.names)
            say('     %-12s weak coeff %+11.4e  SE %9.2e  t %7.1f | trajectory OLS %+11.4e  block-bootstrap SE %9.2e\n', Rw.names{j}, Rw.b(j), Rw.se_all(j), Rw.b(j)/Rw.se_all(j), T(j).coeff, T(j).se_boot);
        end
    end
    say('\nResidual predictability after the weak-form model (held-out NN R^2, one-sided p)\n');
    for cs = {'nominal','no_duffing','no_gyro','aniso','hysteresis','params'}
        S = load(fullfile(RES,'cache',sprintf('%s_31.mat', cs{1}))); D = S.D;
        for e = {'x','y'}
            R = regression_baselines(D, e{1}, 'B', 20);
            Rw = weak_form(D, e{1}, 'terms', R.weak.selected);
            P0 = diagnostics('predictability', D, e{1}, {}, []);
            P1 = diagnostics('predictability', D, e{1}, R.weak.selected, Rw.b);
            say('  %-10s %s-eq: seed only R2 = %6.3f (p = %.3g); seed + selected R2 = %6.3f (p = %.3g)\n', cs{1}, e{1}, P0.R2_heldout, P0.p, P1.R2_heldout, P1.p);
        end
    end
case 'noise'
    say('Weak-form selection versus measurement noise (seed 31 trajectory, 5 realizations each)\n');
    rows = {};
    for c = 1:numel(cases)
        S = load(fullfile(RES,'cache',sprintf('%s_31.mat', cases{c}))); D0 = S.D;
        for nf = [0.005 0.02 0.05 0.10]
            for e = 1:2
                eq = 'xy'; eq = eq(e); v = zeros(1,4);
                for s = 201:205
                    D = simulate_rotor(D0, 'renoise', s, nf);
                    R = regression_baselines(D, eq, 'B', 20); sel = R.weak.selected;
                    v = v + [any(strcmp(sel, mech{e,1})), any(strcmp(sel, mech{e,2})), any(strcmp(sel, mech{e,3})), numel(sel)];
                end
                v = v/5;
                say('  %-10s noise %4.1f%% %s-eq: gyro %.1f cubic %.1f cross %.1f mean# %.1f\n', cases{c}, 100*nf, eq, v);
                rows(end+1,:) = {cases{c}, nf, eq, v(1), v(2), v(3), v(4)}; %#ok<AGROW>
            end
        end
    end
    write_cell(fullfile(RES,'extra_noise.csv'), {'case','noise_frac','eqn','rate_gyro','rate_cubic','rate_cross','mean_n_selected'}, rows);
end
fclose(fid);
end

function D = get_data(case_name, seed)
f = fullfile(result_dir(), 'cache', sprintf('%s_%d.mat', case_name, seed));
if exist(f, 'file'), S = load(f); D = S.D; return; end
if ~exist(fullfile(result_dir(),'cache'),'dir'), mkdir(fullfile(result_dir(),'cache')); end
D = simulate_rotor(spiral_config(case_name, seed));
D = rmfield(D, {'Omega_fun','theta_fun'});
save('-v7', f, 'D');
end

function D = get_test(case_name)
f = fullfile(result_dir(), 'cache', sprintf('%s_test.mat', case_name));
if exist(f, 'file'), S = load(f); D = S.D; return; end
if ~exist(fullfile(result_dir(),'cache'),'dir'), mkdir(fullfile(result_dir(),'cache')); end
cfg = spiral_config(case_name, 77);
cfg.params.phi0 = cfg.params.phi0 + 0.7;    % independent record: other phase and noise
D = simulate_rotor(cfg);
D = rmfield(D, {'Omega_fun','theta_fun'});
save('-v7', f, 'D');
end

function s = describe(D)
c = D.cfg;
s = sprintf('%g s, %g -> %g Hz, dt = %g s, noise %g%%, SG windows %d/%d, N = %d', c.Tend, c.f_start, c.f_end, c.dt, 100*c.noise_frac, D.win_x, D.win_y, numel(D.t));
end

function [g, du, cr] = mechanisms(ox, oy, mode)
% gyroscopic: Omega*ydot in x-eq and Omega*xdot in y-eq
% cubic stiffness (strict): (x^2+y^2)*x, x^3, x^2*y / (x^2+y^2)*y, y^3, x*y^2
% cross-channel: y in x-eq and x in y-eq
if nargin < 3, mode = 'both'; end
gx = any(strcmp(ox.active_names,'Omega*ydot')); gy = any(strcmp(oy.active_names,'Omega*xdot'));
dx = any(ismember({'(x^2+y^2)*x','x^3','x^2*y'}, ox.active_names));
dy = any(ismember({'(x^2+y^2)*y','y^3','x*y^2'}, oy.active_names));
cx = any(strcmp(ox.active_names,'y')); cy = any(strcmp(oy.active_names,'x'));
if strcmp(mode,'any'), g = gx||gy; du = dx||dy; cr = cx||cy;
else, g = gx&&gy; du = dx&&dy; cr = cx&&cy; end
end

function say_both(fid, varargin)
fprintf(varargin{:}); fprintf(fid, varargin{:});
end

function write_cell(fname, header, rows)
fid = fopen(fname, 'w');
fprintf(fid, '%s\n', strjoin(header, ','));
for i = 1:size(rows,1)
    parts = cell(1, size(rows,2));
    for j = 1:size(rows,2)
        v = rows{i,j};
        if ischar(v), parts{j} = ['"' strrep(v,'"','''') '"'];
        elseif islogical(v), parts{j} = sprintf('%d', v);
        else, parts{j} = sprintf('%.6g', v); end
    end
    fprintf(fid, '%s\n', strjoin(parts, ','));
end
fclose(fid);
end

function write_boot(fname, T, o)
fid = fopen(fname, 'w');
fprintf(fid, 'term,coeff,se_boot,se_ols,ci95_lo,ci95_hi,z,p_boot,block_len,B,n_blocks,iter_admitted,gate2_folds_passed,gate2_F,gate2_df2,gate2_p\n');
for j = 1:numel(T)
    k = find(strcmp({o.terms.name}, T(j).name));
    if isempty(k), extra = 'seed,,,,'; else
        tk = o.terms(k); extra = sprintf('%d,%.3f,%.4g,%d,%.4g', tk.iter, tk.gate2_frac, tk.gate2_F, tk.gate2_df, tk.gate2_p); end
    fprintf(fid, '"%s",%.6g,%.4g,%.4g,%.6g,%.6g,%.3f,%.4g,%d,%d,%d,%s\n', T(j).name, T(j).coeff, T(j).se_boot, ...
        T(j).se_ols, T(j).ci_lo, T(j).ci_hi, T(j).z, T(j).p_boot, T(j).block_len, T(j).B, T(j).n_blocks, extra);
end
fclose(fid);
end

function write_cond(fname, C)
fid = fopen(fname, 'w');
fprintf(fid, 'set,kappa,mu_Phi,rank,n_columns,singular_values,columns\n');
fprintf(fid, 'full,%.4g,%.4g,%d,%d,"%s","%s"\n', C.full.cond, C.full.mu_Phi, C.full.rank, numel(C.full.names), mat2str(C.full.sv,4), strjoin(C.full.names,' ; '));
fprintf(fid, 'representatives,%.4g,%.4g,%d,%d,"%s","%s"\n', C.reps.cond, C.reps.mu_Phi, C.reps.rank, numel(C.reps.names), mat2str(C.reps.sv,4), strjoin(C.reps.names,' ; '));
fclose(fid);
end

function d = result_dir()
% results/ folder of the package (created if needed)
d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results');
if ~exist(d, 'dir'), mkdir(d); end
end
