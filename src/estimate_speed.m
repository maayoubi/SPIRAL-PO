function S = estimate_speed(t, x, varargin)
% ESTIMATE_SPEED  Spin speed of a coast-down estimated from one
% displacement record by order tracking (Sect. 5.2 of the paper).
%
%   S = estimate_speed(t, x)                 % x in metres, t in seconds
%   S = estimate_speed(t, x, 'fmax', 70)
%
% Method
%   1. Short-time Fourier transform (Hamming window, 'nperseg' samples,
%      hop 'hop'), log power up to 'fmax' Hz.
%   2. The per-bin median log power over time is subtracted to remove
%      stationary content.
%   3. Ridge tracking by dynamic programming with a penalty 'lambda' on
%      frequency jumps, a cap 'maxdrop_hz_per_s' and (default) a
%      non-increasing frequency.
%   4. The ridge (quantized to the bin width fs/nperseg) is interpolated
%      linearly through the center of each constant-frequency step.
%
% Output S
%   Omega          smoothed speed on t [rad/s] (step-center interpolation)
%   Omega_ridge    raw ridge interpolated on t [rad/s]
%   t_frame, f_ridge   STFT frame times [s] and ridge frequency [Hz]
%   t_step, f_step     step centers used for the smooth curve

p = inputParser;
p.addParameter('fmax', 70.0);
p.addParameter('nperseg', 2048);
p.addParameter('hop', 64);
p.addParameter('lambda', 0.05);
p.addParameter('maxdrop_hz_per_s', 8.0);   % cap on |dOmega/dt|/(2*pi)
p.addParameter('monotonic', true);
p.addParameter('suppress_background', true);
p.parse(varargin{:});
o = p.Results;

dt = t(2) - t(1);
fs = 1/dt;
N  = numel(x);
x0 = x - mean(x);

win = hamming(o.nperseg);
noverlap = o.nperseg - o.hop;
step = o.nperseg - noverlap;
starts = 1:step:(N - o.nperseg + 1);
M = numel(starts);
nfft = o.nperseg;
freqs = (0:nfft/2)' * fs / nfft;

band = freqs <= o.fmax;
f_b  = freqs(band);
nF   = numel(f_b);

logP = zeros(nF, M);
t_frame = zeros(M,1);
for m = 1:M
    seg = x0(starts(m):starts(m)+o.nperseg-1) .* win;
    X = fft(seg, nfft);
    P = abs(X(1:nfft/2+1)).^2;
    logP(:,m) = log(P(band) + 1e-12);
    t_frame(m) = (starts(m) + o.nperseg/2 - 1) * dt;
end

% --- remove stationary content (per-bin median, clipped at zero) ---
if o.suppress_background
    bg = median(logP, 2);
    logP = max(logP - bg, 0);
end

% --- ridge tracking ---
frame_dt = step * dt;
df_bin   = f_b(2) - f_b(1);
maxbins  = max(1, round(o.maxdrop_hz_per_s * frame_dt / df_bin));

INF = 1e18;
cost = INF*ones(nF, M);
back = zeros(nF, M);
cost(:,1) = -logP(:,1);
for ti = 2:M
    for fi = 1:nF
        if o.monotonic
            lo = fi; hi = min(nF, fi + maxbins);
        else
            lo = max(1, fi - maxbins); hi = min(nF, fi + maxbins);
        end
        best = INF; bi = lo;
        for fj = lo:hi
            df = f_b(fi) - f_b(fj);
            pen = o.lambda * df^2;
            c = cost(fj, ti-1) + pen;
            if c < best
                best = c; bi = fj;
            end
        end
        cost(fi,ti) = best - logP(fi,ti);
        back(fi,ti) = bi;
    end
end
path = zeros(M,1);
[~, path(M)] = min(cost(:,M));
for ti = M-1:-1:1
    path(ti) = back(path(ti+1), ti+1);
end
f_ridge = f_b(path);

% --- ridge on the sample times ---
f_full = interp1(t_frame, f_ridge, t, 'linear', 'extrap');
f_full = max(f_full, 0);   % spin speed cannot be negative

Omega_ridge = 2*pi*f_full;

% --- smooth speed: interpolate through the center of each ridge step ---
fr = f_ridge(:); tf = t_frame(:);
edges = [1; find(diff(fr) ~= 0) + 1; numel(fr) + 1];
tc = zeros(numel(edges)-1,1); fc = tc;
for k = 1:numel(edges)-1
    s = edges(k):edges(k+1)-1;
    tc(k) = mean(tf(s)); fc(k) = fr(s(1));
end
S.Omega = max(2*pi*interp1(tc, fc, t(:), 'linear', 'extrap'), 0);
S.Omega_ridge = Omega_ridge;
S.t_frame = t_frame; S.f_ridge = f_ridge;
S.t_step = tc; S.f_step = fc;
end

function w = hamming(N)
% Hamming window
n = (0:N-1)';
w = 0.54 - 0.46*cos(2*pi*n/(N-1));
end
