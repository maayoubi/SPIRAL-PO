function [xd, xdd, win, cv_table] = sg_derivative(x, dt, win, order)
% SG_DERIVATIVE  Savitzky-Golay estimates of the first and second time
% derivative of a sampled signal (Sect. 2.3 of the paper).
%
%   [xd, xdd, win] = sg_derivative(x, dt, 31)          % fixed window
%   [xd, xdd, win] = sg_derivative(x, dt, [21 31 51])  % window chosen by
%                                                      % hold-out cross-validation
%
% Inputs
%   x      signal (N x 1)
%   dt     sample spacing [s]
%   win    window length in samples (odd), or a vector of candidate
%          windows; then every 10th sample is withheld, the local
%          polynomial is fitted to the rest, and the window with the
%          smallest error on the withheld displacement samples is used
%   order  local polynomial order (default 4)
%
% Outputs
%   xd, xdd    first and second derivative (N x 1)
%   win        window actually used
%   cv_table   [window, hold-out RMSE] for each candidate (empty if win
%              was a scalar)

if nargin < 4 || isempty(order), order = 4; end
x = x(:);
cv_table = [];
if numel(win) > 1
    [win, cv_table] = select_window(x, win, order);
end
if mod(win,2) == 0, win = win+1; end

N = numel(x);
h = floor(win/2);
n = (-h:h)';
P = zeros(numel(n), order+1);
for j = 0:order, P(:,j+1) = n.^j; end
Pinv = (P'*P) \ P';
c1 = Pinv(2,:)' / dt;            % first-derivative filter
c2 = (2*Pinv(3,:)') / dt^2;      % second-derivative filter
xpad = [x(1)*ones(h,1); x; x(end)*ones(h,1)];
xd  = conv(xpad, flipud(c1), 'valid');
xdd = conv(xpad, flipud(c2), 'valid');
assert(numel(xd) == N);
end

% -------------------------------------------------------------------------
function [win_best, cv_table] = select_window(x, win_grid, order)
% Hold-out cross-validation of the window length: withhold every 10th
% sample, smooth the rest, score the reconstruction of the withheld samples.
stride = 10;
N = numel(x);
mask = false(N,1); mask(stride:stride:N) = true;
idx_obs = find(~mask);
cv_table = zeros(numel(win_grid), 2);
for k = 1:numel(win_grid)
    w = win_grid(k); if mod(w,2) == 0, w = w+1; end
    h = floor(w/2); n = (-h:h)';
    P = zeros(numel(n), order+1);
    for j = 0:order, P(:,j+1) = n.^j; end
    Pinv = (P'*P) \ P';
    c0 = Pinv(1,:)';
    x_train = x;
    x_train(mask) = interp1(idx_obs, x(idx_obs), find(mask), 'linear', 'extrap');
    xpad = [x_train(1)*ones(h,1); x_train; x_train(end)*ones(h,1)];
    x_smooth = conv(xpad, flipud(c0), 'valid');
    err = x_smooth(mask) - x(mask);
    cv_table(k,:) = [w, sqrt(mean(err.^2))];
end
[~, ibest] = min(cv_table(:,2));
win_best = cv_table(ibest, 1);
end
