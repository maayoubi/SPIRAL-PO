function R = study_rig(part, varargin)
% STUDY_RIG  Experimental part of the paper (Sect. 5).
%
%   study_rig                      % everything below except the audit
%   study_rig('bivariate')         % Table 12: simultaneous x-y record,
%                                  % speed check, tracking robustness,
%                                  % synthetic speed-error control
%   study_rig('figure_bivariate')  % Fig. 6 (spectrograms with speed curves)
%   study_rig('controls')          % synthetic calibration of the single-
%                                  % channel library (Sect. 5.4)
%   study_rig('identify')          % Table 13: single-sensor coast-downs, fit
%                                  % on X and on Y, held-out check
%   study_rig('figure')            % spectrogram/envelopes of the single-sensor runs
%   study_rig('audit', raw_dir)    % provenance of the raw sensor files
%
% Two kinds of rig record (data/):
%   Experimental_Rotor_Data_SPIRAL_PO.csv  x and y recorded simultaneously
%       by two laser sensors (Quanser DAQ, 1 kHz), coast-down 32 Hz -> 0 in
%       6.7 s. The speed column assumes a constant deceleration; the speed
%       is estimated by order tracking (estimate_speed.m).
%   SIG_9_9_2022.mat  single-sensor coast-downs from a separate session
%       (ILD1420 at 4 kHz, one direction per run, 60 -> 3 Hz), single-channel
%       model
%         sdd = seed + b1 s^3 + b2 s^2 sd + b3 Omega^2 s + b4 Omegadot s,
%       weak form, Bonferroni over 4 candidates (|t| > 2.50); held-out check
%       on the other runs with the physical coefficients fixed and only the
%       unbalance amplitude and phase refitted.
%
% Output (results/): rig_bivariate.txt, Fig_rig_simultaneous.pdf/.png,
% rig_single_channel_fitX.txt/.csv, rig_single_channel_fitY.txt/.csv,
% single_channel_controls.txt, Fig_rig_single_channel.pdf/.png,
% rig_audit.txt

if nargin < 1, part = 'all'; end
RES = result_dir();
R = struct();
switch lower(part)
    case 'all'
        R.bivariate = rig_bivariate(RES);
        rig_figure_bivariate(RES);
        R.controls = rig_controls(RES);
        R.identify = rig_identify(RES);
        rig_figure(RES);
    case 'controls', R = rig_controls(RES);
    case 'identify', R = rig_identify(RES);
    case 'figure',   rig_figure(RES);
    case 'audit',    rig_audit(RES, varargin{:});
    case 'bivariate', R = rig_bivariate(RES);
    case 'figure_bivariate', rig_figure_bivariate(RES);
    otherwise, error('study_rig:part', 'unknown part %s', part);
end
end

% ========================================================================
function out = rig_bivariate(RES)
% Simultaneous two-sensor coast-down: (1) tracked speed versus the
% constant-deceleration line, (2) bivariate weak form with the tracked
% speed, (3) nine tracking settings, (4) constant-deceleration line,
% (5) synthetic speed-error control.
fid = fopen(fullfile(RES, 'rig_bivariate.txt'), 'w');
say = @(varargin) say_both(fid, varargin{:});
T = load_table();
t = T.t; dt = t(2)-t(1);
say('Simultaneous x-y coast-down: %d samples at %.0f Hz (%.2f s); speed column %.1f -> %.1f Hz (straight line)\n', ...
    numel(t), 1/dt, t(end), T.Om(1)/2/pi, T.Om(end)/2/pi);

% (1) speed: tracked ridge versus the constant-deceleration line
SX = estimate_speed(t, T.x, 'fmax', 40, 'nperseg', 512, 'hop', 16, 'maxdrop_hz_per_s', 15);
SY = estimate_speed(t, T.y, 'fmax', 40, 'nperseg', 512, 'hop', 16, 'maxdrop_hz_per_s', 15);
Om = (SX.Omega + SY.Omega)/2;
act = Om > 2*pi*3;
dev = (T.Om - Om)/(2*pi);
say('\n(1) Speed column minus order-tracked speed (1.95 Hz bins, x and y averaged), where tracked speed > 3 Hz:\n');
say('    mean %.2f Hz, RMS %.2f Hz, max %.2f Hz\n', mean(dev(act)), sqrt(mean(dev(act).^2)), max(abs(dev(act))));
for tq = 0.5:1:5.5
    [~, i] = min(abs(t - tq));
    say('    t = %.1f s: tracked %5.1f Hz, speed column %5.1f Hz\n', tq, Om(i)/2/pi, T.Om(i)/2/pi);
end
% spectral power along each speed curve (1x, 2x) above the median background
for ch = {'x','y'}
    pw = harmonic_power(t, T.(ch{1}), Om/(2*pi), T.Om/(2*pi), 512);
    say('    %s: power above background along tracked / line: 1x %.1f / %.1f dB, 2x %.1f / %.1f dB\n', ch{1}, pw);
end
[f, C] = coherence(T.x, T.y, 1/dt, 1024);
say('    x-y magnitude-squared coherence: max %.3f at %.1f Hz\n', max(C(2:end)), f(1+find(C(2:end) == max(C(2:end)), 1)));

% (2) bivariate weak form with the tracked speed
D = make_record(T, Om);
say('\n(2) Weak-form identification, tracked speed (%.2f s used, %d test functions per equation)\n', D.t(end), weak_form(D,'x').n_eq);
for e = {'x','y'}
    W = weak_form(D, e{1}); R = regression_baselines(D, e{1}, 'B', 20);
    say('    %s-eq t:', e{1});
    for j = 1:numel(W.names), say(' %s %.1f |', W.names{j}, W.t(j)); end
    say('\n        selected (|t| > %.2f): %s\n        pointwise OLS (classical SE) would select: %s\n', W.crit, ...
        strjoin(W.selected, ', '), strjoin(R.ols_naive.selected, ', '));
    out.(['W' e{1}]) = W;
end

% (3) robustness over tracking settings
say('\n(3) Robustness: window 256/512/1024 samples x speed from x, y or both\n');
cnt = containers.Map();
for np_ = [256 512 1024]
    SXn = estimate_speed(t, T.x, 'fmax', 40, 'nperseg', np_, 'hop', 16, 'maxdrop_hz_per_s', 15);
    SYn = estimate_speed(t, T.y, 'fmax', 40, 'nperseg', np_, 'hop', 16, 'maxdrop_hz_per_s', 15);
    srcs = {SXn.Omega, SYn.Omega, (SXn.Omega+SYn.Omega)/2}; lab = {'x','y','both'};
    for s = 1:3
        Dn = make_record(T, srcs{s});
        Wx = weak_form(Dn, 'x'); Wy = weak_form(Dn, 'y');
        say('    %4d %-4s | x: %-32s | y: %s\n', np_, lab{s}, strjoin(Wx.selected, ', '), strjoin(Wy.selected, ', '));
        for nm = [Wx.selected, Wy.selected]
            if isKey(cnt, nm{1}), cnt(nm{1}) = cnt(nm{1}) + 1; else, cnt(nm{1}) = 1; end
        end
    end
end
say('    selection counts over 9 settings: ');
k = keys(cnt); for j = 1:numel(k), say('%s %d/9; ', k{j}, cnt(k{j})); end
say('\n');

% (4) the constant-deceleration line
Dl = make_record(T, T.Om, Om > 2*pi*3);
say('\n(4) Weak form with the speed column (constant-deceleration line)\n');
for e = {'x','y'}
    W = weak_form(Dl, e{1});
    say('    %s-eq selected: %s\n', e{1}, strjoin(W.selected, ', '));
end

% (5) synthetic control: a speed line 4 Hz too high at the start
say('\n(5) Synthetic control: exact speed / bin-limited tracked speed / straight line starting 4 Hz too high\n');
for cs = {'nominal','no_gyro','no_duffing'}
    Ds = simulate_rotor(spiral_config(cs{1}, 31)); ts = Ds.t; Ot = Ds.Omega;
    line = 2*pi*interp1([ts(1) ts(end)], [Ot(1)/2/pi + 4, Ot(end)/2/pi], ts);
    fq = 1.953*round(Ot/(2*pi*1.953));
    edges = [1; find(diff(fq) ~= 0) + 1; numel(fq) + 1]; tc = zeros(numel(edges)-1,1); fc = tc;
    for j = 1:numel(edges)-1, s = edges(j):edges(j+1)-1; tc(j) = mean(ts(s)); fc(j) = fq(s(1)); end
    trk = 2*pi*interp1(tc, fc, ts, 'linear', 'extrap');
    V = {'exact', Ot; 'tracked', trk; 'line +4 Hz', line};
    for v = 1:3
        Dv = Ds; Dv.Omega = V{v,2}; Dv.Omegadot = gradient(Dv.Omega, ts(2)-ts(1));
        Wx = weak_form(Dv, 'x'); Wy = weak_form(Dv, 'y');
        say('    %-10s %-10s x: %-40s y: %s\n', cs{1}, V{v,1}, strjoin(Wx.selected, ', '), strjoin(Wy.selected, ', '));
    end
end
fclose(fid);
out.Omega_tracked = Om; out.table = T;
end

function T = load_table()
M = dlmread(fullfile(data_dir(), 'Experimental_Rotor_Data_SPIRAL_PO.csv'), ',', 1, 0);
T.t = M(:,1) - M(1,1); T.x = M(:,2) - mean(M(:,2)); T.y = M(:,3) - mean(M(:,3)); T.Om = M(:,4);
end

function D = make_record(T, Om, keep)
% Bivariate record from the simultaneous table with speed Om; the part with
% speed above 3 Hz is used.
dt = T.t(2) - T.t(1);
D.t = T.t; D.x = T.x; D.y = T.y; D.Omega = Om(:); D.Omegadot = gradient(D.Omega, dt);
[D.xd, D.xdd, D.win_x] = sg_derivative(T.x, dt, [11 21 31], 4);
[D.yd, D.ydd, D.win_y] = sg_derivative(T.y, dt, [11 21 31], 4);
if nargin < 3, keep = D.Omega > 2*pi*3; end
for f = {'t','x','y','Omega','Omegadot','xd','xdd','yd','ydd'}, D.(f{1}) = D.(f{1})(keep); end
D.t = D.t - D.t(1);
end

function pw = harmonic_power(t, s, f_trk, f_line, nw)
% Median STFT power (dB above the per-frame median background, 2-80 Hz) at
% f_trk, f_line, 2 f_trk and 2 f_line, over frames with tracked speed > 4 Hz.
fs = 1/(t(2)-t(1)); nfft = 8192; w = 0.5 - 0.5*cos(2*pi*(0:nw-1)'/(nw-1));
f = (0:nfft/2)'*fs/nfft; bgb = f > 2 & f < 80;
v = zeros(0,4);
for s0 = 1:32:numel(s)-nw
    tc = t(s0 + nw/2);
    ft = interp1(t, f_trk, tc); fl = interp1(t, f_line, tc);
    if ft < 4 || tc > 5.5, continue; end
    P = abs(fft((s(s0:s0+nw-1) - mean(s)).*w, nfft)).^2; P = P(1:nfft/2+1);
    bg = median(P(bgb));
    at = @(f0) P(1 + round(f0*nfft/fs));
    v(end+1,:) = [at(ft), at(fl), at(2*ft), at(2*fl)]/bg; %#ok<AGROW>
end
pw = 10*log10(median(v, 1));
end

function [f, C] = coherence(a, b, fs, n)
% Welch magnitude-squared coherence (Hann window, 50 % overlap).
a = a - mean(a); b = b - mean(b); w = 0.5 - 0.5*cos(2*pi*(0:n-1)'/(n-1));
Sab = 0; Saa = 0; Sbb = 0;
for s = 1:n/2:numel(a)-n
    A = fft(a(s:s+n-1).*w); B = fft(b(s:s+n-1).*w);
    Sab = Sab + A.*conj(B); Saa = Saa + abs(A).^2; Sbb = Sbb + abs(B).^2;
end
k = 1:n/2+1; f = (k-1)'*fs/n;
C = abs(Sab(k)).^2 ./ (Saa(k).*Sbb(k));
end

function rig_figure_bivariate(RES)
% Spectrogram of the simultaneous record with the speed column, the tracked
% speed and its second harmonic.
T = load_table(); t = T.t; fs = 1/(t(2)-t(1));
SX = estimate_speed(t, T.x, 'fmax', 40, 'nperseg', 512, 'hop', 16, 'maxdrop_hz_per_s', 15);
SY = estimate_speed(t, T.y, 'fmax', 40, 'nperseg', 512, 'hop', 16, 'maxdrop_hz_per_s', 15);
Om = (SX.Omega + SY.Omega)/2;
fig = figure('Units','centimeters','Position',[2 2 18 7],'Color','w');
lab = {'x','y'}; sig = {T.x, T.y};
for p = 1:2
    subplot(1,2,p);
    nwin = 512; hop = 16; win = 0.5 - 0.5*cos(2*pi*(0:nwin-1)'/(nwin-1));
    nfr = floor((numel(sig{p}) - nwin)/hop) + 1; fax = (0:nwin/2)'*fs/nwin; kp = fax <= 70;
    P = zeros(nnz(kp), nfr); ts = zeros(1,nfr);
    for k = 1:nfr
        Fk = abs(fft(sig{p}((k-1)*hop + (1:nwin)).*win)).^2; P(:,k) = Fk(kp); ts(k) = ((k-1)*hop + nwin/2)/fs;
    end
    PdB = 10*log10(P + eps); PdB = PdB - max(PdB(:));
    imagesc(ts, fax(kp), PdB); axis xy; colormap(flipud(gray)); caxis([-50 0]); hold on
    plot(t, T.Om/2/pi, 'r-', 'LineWidth', 1.2);
    plot(t, Om/2/pi, '-', 'Color', [0.13 0.27 0.60], 'LineWidth', 1.2);
    plot(t, 2*Om/2/pi, '--', 'Color', [0.13 0.27 0.60], 'LineWidth', 0.8);
    xlabel('time [s]'); ylabel('frequency [Hz]'); ylim([0 70]);
    title(sprintf('(%s) %s channel', char('a'+p-1), lab{p}), 'FontWeight', 'normal');
    if p == 1, legend({'constant-deceleration speed','order-tracked speed','2 x tracked'}, 'Box','off'); end
end
set(findall(fig, '-property', 'FontSize'), 'FontSize', 9);
print(fig, fullfile(RES, 'Fig_rig_simultaneous.pdf'), '-dpdf', '-painters', '-bestfit');
print(fig, fullfile(RES, 'Fig_rig_simultaneous.png'), '-dpng', '-r200');
end

% ========================================================================
function D = load_record(name, fs_out, t_window)
% Preprocess one lateral rig record ('X' or 'Y') into a single-channel record.
fs = 4000;
S = load(fullfile(data_dir(), 'SIG_9_9_2022.mat'));
raw_mm = double(S.(sprintf('SIG_CH_%s_DECEL_HZ', name))(:));
bad = raw_mm <= 0;                       % ILD1420 out-of-range code (target lost)
if any(bad)
    ii = (1:numel(raw_mm))';
    raw_mm(bad) = interp1(ii(~bad), raw_mm(~bad), ii(bad), 'linear', 'extrap');
end
raw = raw_mm * 1e-3;                     % mm -> m
t4 = (0:numel(raw)-1)'/fs;
SP = estimate_speed(t4, raw - mean(raw));
dec = round(fs/fs_out);
x = decimate_fir(raw - mean(raw), dec);
t = (0:numel(x)-1)'/fs_out;
Om = interp1(t4, SP.Omega, t, 'linear', 'extrap');
keep = t >= t_window(1) & t <= t_window(2) & Om > 2*pi*3;
D.name = name; D.n_dropouts = sum(bad);
D.t = t(keep) - t(find(keep,1)); D.x = x(keep); D.Omega = Om(keep);
D.Omegadot = gradient(D.Omega, 1/fs_out);
[D.xd, D.xdd, D.win] = sg_derivative(D.x, 1/fs_out, [21 31 51], 4);
env = sqrt(movmean(D.x.^2, round(0.25*fs_out)));
[pk, ipk] = max(env);
D.peak_rms = pk; D.f_at_peak = D.Omega(ipk)/(2*pi);
D.t59_to_10 = interp_cross(SP.t_frame, SP.f_ridge, 10) - interp_cross(SP.t_frame, SP.f_ridge, 59);
D.speed = SP;
end

% ========================================================================
function out = rig_identify(RES)
fs_out = 2000; t_window = [1 24];
recs = {'X','Y'};   % lateral runs only; the Z record is axial and outside the planar model
for r = 1:numel(recs), Dall.(recs{r}) = load_record(recs{r}, fs_out, t_window); end
for fitrec = {'X','Y'}
    fid = fopen(fullfile(RES, sprintf('rig_single_channel_fit%s.txt', fitrec{1})), 'w');
    say = @(varargin) say_both(fid, varargin{:});
    say('Single-channel rig identification (raw ILD1420 records, 4000 Hz -> %d Hz)\n', fs_out);
    say('Each record is one direction from a separate run; no bivariate terms are used.\n\n');
    for r = 1:numel(recs)
        D = Dall.(recs{r});
        say('record %s: %d dropout samples replaced, %.1f s used, Omega %.1f -> %.1f Hz, SG window %d, peak RMS %.3f mm at %.1f Hz, 59->10 Hz in %.1f s\n', ...
            recs{r}, D.n_dropouts, D.t(end), D.Omega(1)/2/pi, D.Omega(end)/2/pi, D.win, 1e3*D.peak_rms, D.f_at_peak, D.t59_to_10);
    end
    D = Dall.(fitrec{1});
    W = weak_form(D, 's');
    names = W.names; K = numel(names);
    [S0, L] = columns(D);
    [bo, seo] = ols_se([S0 L], D.xdd);
    to = bo(5:end)./seo(5:end);
    sel = abs(W.t) > W.crit;
    say('\nIdentification on record %s (weak form, %d test functions; Bonferroni |t| > %.2f over %d terms)\n', fitrec{1}, W.n_eq, W.crit, K);
    say('   %-14s %12s %10s %10s %12s\n', 'term', 'coeff', 't (weak)', 't (OLS)', 'selected');
    for j = 1:K
        say('   %-14s %+12.4e %10.2f %10.2f %12d\n', names{j}, W.b(4+j), W.t(j), to(j), sel(j));
    end
    say('   seed: k/m = %.4g 1/s^2 (f_n = %.2f Hz), c/m = %.4g 1/s\n', W.b(4), sqrt(max(W.b(4),0))/2/pi, W.b(3));
    bm = W.G(:, [true(1,4), sel]) \ W.Gy;     % seed + selected, weak form
    bs = W.G(:, 1:4) \ W.Gy;                  % seed only
    say('\nHeld-out check (physical coefficients fixed; unbalance amplitude/phase refitted per record)\n');
    rows = {};
    for r = 1:numel(recs)
        Dt = Dall.(recs{r});
        [St, Lt] = columns(Dt);
        gain = heldout_gain(St, Lt(:, sel), Dt.xdd, bm, bs);
        isfit = strcmp(recs{r}, fitrec{1});
        rows(end+1,:) = {recs{r}, isfit, gain, Dt.peak_rms, Dt.f_at_peak}; %#ok<AGROW>
        say('   record %s%s: share of post-seed residual explained by the selected terms = %6.2f%%\n', ...
            recs{r}, ternary(isfit, ' (fit record)', ''), gain);
    end
    say('\nSelected terms: %s\n', strjoin(names(sel), ', '));
    fclose(fid);
    f2 = fopen(fullfile(RES, sprintf('rig_single_channel_fit%s.csv', fitrec{1})), 'w');
    fprintf(f2, 'record,is_fit_record,heldout_gain_pct,peak_rms_m,f_at_peak_hz\n');
    for i = 1:size(rows,1), fprintf(f2, '%s,%d,%.4f,%.6g,%.3f\n', rows{i,:}); end
    fclose(f2);
    out.(['fit' fitrec{1}]) = struct('names', {names}, 'coeff', W.b, 't_weak', W.t, 't_ols', to', 'selected', {names(sel)});
end
out.records = Dall;
end

% ========================================================================
function out = rig_controls(RES)
% The x channel of the synthetic benchmark analyzed exactly like a rig
% record (one channel, weak form, 4 candidates), with the exact speed and
% with a speed degraded like the rig estimate (1.95 Hz bins, step-center
% interpolation).
fid = fopen(fullfile(RES, 'single_channel_controls.txt'), 'w');
crit = sqrt(2)*erfinv(2*(1-0.05/8)-1);
say_both(fid, 'Single-channel weak-form |t| (Bonferroni threshold %.2f)\n', crit);
out = {};
for cs = {'nominal','no_gyro','no_duffing'}
    Ds = simulate_rotor(spiral_config(cs{1}, 31));
    for mode = {'exact','tracked'}
        D = struct('t', Ds.t, 'x', Ds.x, 'xd', Ds.xd);
        Om = Ds.Omega;
        if strcmp(mode{1}, 'tracked')
            fq = 1.953*round(Om/(2*pi*1.953));
            edges = [1; find(diff(fq) ~= 0) + 1; numel(fq) + 1];
            tc = zeros(numel(edges)-1,1); fc = tc;
            for k = 1:numel(edges)-1, s = edges(k):edges(k+1)-1; tc(k) = mean(Ds.t(s)); fc(k) = fq(s(1)); end
            Om = 2*pi*interp1(tc, fc, Ds.t, 'linear', 'extrap');
        end
        D.Omega = Om; D.Omegadot = gradient(Om, Ds.t(2)-Ds.t(1));
        W = weak_form(D, 's');
        say_both(fid, '%-11s %-8s t: s^3 %6.1f  s^2*sdot %6.1f  Omega^2*s %6.1f  Omegadot*s %6.1f  -> selected: %s\n', ...
            cs{1}, mode{1}, W.t, strjoin(W.names(abs(W.t) > crit), ', '));
        out(end+1,:) = {cs{1}, mode{1}, W.t, W.names(abs(W.t) > crit)}; %#ok<AGROW>
    end
end
fclose(fid);
end

% ========================================================================
function rig_figure(RES)
% Single-sensor runs: (a) spectrogram of record X with the STFT ridge and the smoothed
% speed; (b) RMS envelope (0.25 s) versus estimated speed for X and Y.
fs = 4000;
S = load(fullfile(data_dir(), 'SIG_9_9_2022.mat'));
rec = {'X','Y'}; col = {[0.13 0.27 0.60], [0.93 0.49 0.13]};
for r = 1:2
    raw = double(S.(sprintf('SIG_CH_%s_DECEL_HZ', rec{r}))(:));
    bad = raw <= 0;
    if any(bad)
        ii = (1:numel(raw))';
        raw(bad) = interp1(ii(~bad), raw(~bad), ii(bad), 'linear', 'extrap');
    end
    x = raw - mean(raw);                         % mm
    t = (0:numel(x)-1)'/fs;
    SP = estimate_speed(t, x*1e-3);
    F.(rec{r}) = struct('t', t, 'x', x, 'f', SP.Omega/(2*pi), 'tf', SP.t_frame, 'fr', SP.f_ridge, ...
                        'env', sqrt(movmean(x.^2, round(0.25*fs))));
end
fig = figure('Units', 'centimeters', 'Position', [2 2 18 7], 'Color', 'w');
ax1 = subplot(1, 2, 1);
X = F.X; nwin = 2048; hop = 256;
win = 0.5 - 0.5*cos(2*pi*(0:nwin-1)'/(nwin-1));
nfr = floor((numel(X.x) - nwin)/hop) + 1;
fax = (0:nwin/2)'*fs/nwin; keep = fax <= 80;
P = zeros(nnz(keep), nfr); ts = zeros(1, nfr);
for k = 1:nfr
    seg = X.x((k-1)*hop + (1:nwin)) .* win;
    Fk = abs(fft(seg)).^2; P(:,k) = Fk(keep); ts(k) = ((k-1)*hop + nwin/2)/fs;
end
PdB = 10*log10(P + eps); PdB = PdB - max(PdB(:));
imagesc(ts, fax(keep), PdB); axis xy; colormap(ax1, flipud(gray)); caxis([-60 0]); hold on
stairs(X.tf, X.fr, '-', 'Color', [0.93 0.49 0.13], 'LineWidth', 0.8);
plot(X.t(1:200:end), X.f(1:200:end), '-', 'Color', [0.75 0.15 0.15], 'LineWidth', 1.4);
text(2, 27, 'stationary 16.6 Hz component', 'FontSize', 8, 'VerticalAlignment', 'bottom');
quiver(3, 26.5, 1, -9.5, 0, 'Color', 'k', 'MaxHeadSize', 0.8, 'LineWidth', 0.6);
xlabel('time [s]'); ylabel('frequency [Hz]'); ylim([0 80]); xlim([0 ts(end)]);
title('(a) record X: spectrogram and order-tracked speed', 'FontWeight', 'normal');
legend({'STFT ridge (2 Hz bins)', 'smoothed speed estimate'}, 'Location', 'northeast', 'Box', 'off');
ax2 = subplot(1, 2, 2); hold on; box on; grid on
for r = 1:2
    d = F.(rec{r}); idx = find(d.f > 3 & d.t > 0.5); idx = idx(1:100:end);
    plot(d.f(idx), d.env(idx), '-', 'Color', col{r}, 'LineWidth', 1.1);
end
set(ax2, 'XDir', 'reverse'); xlim([3 62]);
xlabel('estimated spin frequency [Hz]'); ylabel('RMS displacement [mm]');
title('(b) response envelope, two separate runs', 'FontWeight', 'normal');
legend({'record X', 'record Y'}, 'Location', 'northwest', 'Box', 'off');
set(findall(fig, '-property', 'FontSize'), 'FontSize', 9);
print(fig, fullfile(RES, 'Fig_rig_single_channel.pdf'), '-dpdf', '-painters', '-bestfit');
print(fig, fullfile(RES, 'Fig_rig_single_channel.png'), '-dpng', '-r200');
end

% ========================================================================
function rig_audit(RES, raw_dir)
% Consistency check of the rig data files against the raw sensor exports.
if nargin < 2, raw_dir = fullfile(fileparts(fileparts(data_dir())), 'Experimental Data'); end
diary(fullfile(RES, 'rig_audit.txt')); cleanup = onCleanup(@() diary('off'));
L = dir(fullfile(raw_dir, 'Sample_*.csv'));
if isempty(L), error('study_rig:noFiles', 'no Sample_*.csv files in %s', raw_dir); end
fprintf('=== 1. Raw sensor records (%s) ===\n', raw_dir);
fprintf('%-42s %-10s %7s %9s %11s %11s %8s %9s\n', 'file', 'serial', 'rate', 'N', 't_first[s]', 't_last[s]', 'dur[s]', 'mean[mm]');
Rr = cell(numel(L),1);
for k = 1:numel(L)
    Rr{k} = load_ild_csv(fullfile(raw_dir, L(k).name));
    fprintf('%-42s %-10s %6.0fHz %9d %11.3f %11.3f %8.2f %9.3f\n', L(k).name, Rr{k}.serial, Rr{k}.rate_hz, ...
        Rr{k}.N, Rr{k}.t_first, Rr{k}.t_last, Rr{k}.duration, mean(Rr{k}.d));
end
serials = cellfun(@(r) r.serial, Rr, 'UniformOutput', false);
fprintf('\nDistinct controller serial numbers: %s\n', strjoin(unique(serials), ', '));
ov = 0;
for i = 1:numel(Rr)
    for j = i+1:numel(Rr)
        if Rr{i}.t_first < Rr{j}.t_last && Rr{j}.t_first < Rr{i}.t_last, ov = ov + 1; end
    end
end
fprintf('Pairs of records that overlap in time (same DAQ clock): %d\n', ov);
if numel(unique(serials)) == 1 && ov == 0
    fprintf('--> All records come from ONE sensor and none overlap in time: no two\n    channels were measured simultaneously.\n');
end
fprintf('\n=== 2. Channels of SIG_9_9_2022.mat ===\n');
S = load(fullfile(data_dir(), 'SIG_9_9_2022.mat'));
for f = fieldnames(S)'
    v = S.(f{1}); if numel(v) < 100, continue; end
    hit = '';
    for k = 1:numel(Rr)
        if numel(v) == Rr{k}.N && max(abs(v(:) - Rr{k}.d(:))) < 1e-9, hit = [L(k).name ' (identical)']; break; end
        if numel(v) < Rr{k}.N
            for s = find(abs(Rr{k}.d - v(1)) < 1e-12)'
                if s+numel(v)-1 <= Rr{k}.N && max(abs(Rr{k}.d(s:s+numel(v)-1) - v(:))) < 1e-9
                    hit = sprintf('%s, samples %d-%d', L(k).name, s, s+numel(v)-1); break;
                end
            end
            if ~isempty(hit), break; end
        end
    end
    if isempty(hit), hit = 'no exact match'; end
    fprintf('%-20s N=%7d  <- %s\n', f{1}, numel(v), hit);
end
fprintf('\n=== 3. Experimental_Rotor_Data_SPIRAL_PO (simultaneous coast-down table) ===\n');
M = dlmread(fullfile(data_dir(), 'Experimental_Rotor_Data_SPIRAL_PO.csv'), ',', 1, 0);
t = M(:,1); X = M(:,2); Y = M(:,3); Om = M(:,4);
fprintf('N = %d, dt = %.4g s (%.0f Hz), duration %.2f s\n', numel(t), t(2)-t(1), 1/(t(2)-t(1)), t(end)-t(1));
fprintf('Omega: %.2f -> %.2f rad/s; max |2nd difference| = %.3g rad/s\n', Om(1), Om(end), max(abs(diff(Om,2))));
if max(abs(diff(Om,2))) < 1e-9*max(abs(Om))
    fprintf('--> Omega is an exact straight line: constructed, not measured.\n');
end
fs_tab = 1/(t(2)-t(1));
for ch = {'X','Y','Z'}
    nm = sprintf('SIG_CH_%s_DECEL_HZ', ch{1});
    if ~isfield(S, nm), continue; end
    v = S.(nm)(:); v = v - mean(v); vd = v(1:round(4000/fs_tab):end);
    [cx, lx] = best_corr(vd, X); [cy, ly] = best_corr(vd, Y);
    fprintf('raw %s decel vs table: max |corr| with X_m = %.3f (lag %d), with Y_m = %.3f (lag %d)\n', ch{1}, cx, lx, cy, ly);
end
fprintf('(Values far below 1 mean the table is not a resampled copy of these raw records.)\n');
end

% ========================================================================
function [S, L] = columns(D)
th = cumtrapz(D.t, D.Omega);
S = [D.Omega.^2.*cos(th), D.Omega.^2.*sin(th), -D.xd, -D.x];
L = [D.x.^3, D.x.^2.*D.xd, D.Omega.^2.*D.x, D.Omegadot.*D.x];
end

function g = heldout_gain(S, Lsel, y, bm, bs)
% unbalance coefficients (columns 1-2) refitted on the test record,
% physical coefficients (columns 3-end) kept from the fit record
yp = y - [S(:,3:4), Lsel]*bm(3:end); af = S(:,1:2) \ yp;
e_full = yp - S(:,1:2)*af;
yp0 = y - S(:,3:4)*bs(3:4); a0 = S(:,1:2) \ yp0;
e_seed = yp0 - S(:,1:2)*a0;
g = 100*(1 - sum(e_full.^2)/sum(e_seed.^2));
end

function [b, se] = ols_se(P, y)
sc = sqrt(mean(P.^2,1)); sc(sc==0) = 1; Ps = P./sc;
bs = Ps\y; res = y - Ps*bs; n = size(P,1); q = size(P,2);
s2 = sum(res.^2)/(n-q); b = bs./sc(:); se = sqrt(diag(s2*inv(Ps'*Ps)))./sc(:);
end

function y = decimate_fir(x, r)
% Windowed-sinc low-pass (cut-off 0.8 x new Nyquist), then keep every r-th sample.
if r == 1, y = x; return; end
n = 64*r; k = (-n:n)'; fc = 0.8/(2*r);
s = ones(size(k)); i = k ~= 0; s(i) = sin(pi*2*fc*k(i))./(pi*2*fc*k(i));
h = 2*fc*s .* (0.54 + 0.46*cos(pi*k/n));
h = h/sum(h);
xf = conv(x, h, 'same');
y = xf(1:r:end);
end

function tc = interp_cross(t, f, level)
i = find(f <= level, 1);
if isempty(i) || i == 1, tc = NaN; return; end
tc = t(i-1) + (level - f(i-1))*(t(i)-t(i-1))/(f(i)-f(i-1));
end

function Rr = load_ild_csv(fname)
% Read one Micro-Epsilon optoNCDT ILD1420 CSV export.
fid = fopen(fname, 'r');
if fid < 0, error('study_rig:open', 'cannot open %s', fname); end
h1 = fgetl(fid); h2 = fgetl(fid); fgetl(fid);
fclose(fid);
Rr.file = fname;
Rr.serial = header_field(h1, 'Controller serial number:');
Rr.rate_hz = 1000*str2double(regexprep(header_field(h1, 'Measurement rate:'), '[^0-9.]', ''));
Rr.daq_start = strtrim(strrep(strrep(h2, 'DAQ-Start-Time:', ''), ',', ' '));
M = dlmread(fname, ',', 3, 0);
Rr.t = M(:,1); Rr.d = M(:,2);
Rr.N = numel(Rr.t); Rr.t_first = Rr.t(1); Rr.t_last = Rr.t(end);
Rr.duration = Rr.t_last - Rr.t_first;
end

function v = header_field(line, key)
i = strfind(line, key);
if isempty(i), v = ''; return; end
rest = line(i+numel(key):end);
j = find(rest == ',', 1); if isempty(j), j = numel(rest)+1; end
v = strtrim(rest(1:j-1));
end

function [c, lag] = best_corr(v, w)
n = numel(w); w = (w - mean(w))/std(w);
c = 0; lag = 0;
cs = cumsum([0; v]); cs2 = cumsum([0; v.^2]);
for k = 1:numel(v)-n+1
    mu = (cs(k+n)-cs(k))/n; sd = sqrt(max((cs2(k+n)-cs2(k))/n - mu^2, eps));
    r = abs(sum((v(k:k+n-1)-mu).*w))/(n*sd);
    if r > c, c = r; lag = k-1; end
end
end

function s = ternary(c,a,b), if c, s = a; else, s = b; end, end
function say_both(fid, varargin), fprintf(varargin{:}); fprintf(fid, varargin{:}); end

function d = data_dir()
d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
end

function d = result_dir()
d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results');
if ~exist(d, 'dir'), mkdir(d); end
end
