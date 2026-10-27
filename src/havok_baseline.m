function H = havok_baseline(D_fit, D_test, k_steps, varargin)
% HAVOK_BASELINE  Delay-coordinate (HAVOK-type) baseline on x, y only
% (Brunton et al. 2017; Sect. 4.9 and Table 5 of the paper). A backward-looking
% Hankel matrix of [x; y] is built from D_fit, its leading r right
% singular vectors define delay coordinates v, and a linear model
%   dv_{1:r-1}/dt = A v_{1:r-1} + B v_r
% is fitted by least squares with v_r as the intermittent forcing. The
% model predicts x and y k steps ahead on D_test in two ways:
%   'forced' : v_r taken from the measured test data over the horizon
%   'frozen' : v_r held at its value at the start of the horizon
%
%   H = havok_baseline(D, D2, 20, 'q', 50, 'stride', 4, 'r', 15)
%
% Output: H.nrmse_forced_x/y, H.nrmse_frozen_x/y, H.r, H.q, H.stride,
% H.sv (leading singular values), H.n_params

p = inputParser;
p.addParameter('q', 50);        % delays per channel
p.addParameter('stride', 4);    % delay spacing in samples (4 -> 2 ms)
p.addParameter('r', 15);        % rank (r-1 coordinates + 1 forcing)
p.addParameter('eval_stride', 50);
p.parse(varargin{:});
o = p.Results;
dt = D_fit.t(2) - D_fit.t(1);

[Hf, j0] = hankel_xy(D_fit, o.q, o.stride);
mu = mean(Hf, 2);
[U, S, V] = svd(Hf - mu, 'econ');
r = o.r;
Ur = U(:,1:r); Sr = diag(S(1:r,1:r));
v = V(:,1:r);                                  % M x r, columns ~ unit norm
% derivatives of the delay coordinates (4th-order central differences)
dv = (-v(5:end,:) + 8*v(4:end-1,:) - 8*v(2:end-3,:) + v(1:end-4,:)) / (12*dt);
vv = v(3:end-2,:);
Xi = vv \ dv(:,1:r-1);                          % (r x r-1): [A; B]
A = Xi(1:r-1,:)'; B = Xi(r,:)';

% test coordinates through the training basis
[Ht, jt0] = hankel_xy(D_test, o.q, o.stride);
vt = ((Ur' * (Ht - mu)) ./ Sr)';               % Mt x r
Mt = size(vt,1);
starts = 1:o.eval_stride:(Mt - k_steps);
Ad = expm_local(A*dt);
ex_f = []; ey_f = []; ex_z = []; ey_z = [];
xrow = 1; yrow = o.q + 1;                      % current-time rows of x and y
for s0 = starts
    zf = vt(s0,1:r-1)'; zz = zf;
    for k = 1:k_steps
        uf = vt(s0+k-1, r); uz = vt(s0, r);
        zf = rk4_lin(A, B, zf, uf, vt(s0+k, r), dt);
        zz = rk4_lin(A, B, zz, uz, uz, dt);
    end
    hf = mu + Ur * ([zf; vt(s0+k_steps, r)] .* Sr);
    hz = mu + Ur * ([zz; vt(s0, r)] .* Sr);
    i_true = jt0 + s0 + k_steps - 1;
    ex_f(end+1) = hf(xrow) - D_test.x(i_true); ey_f(end+1) = hf(yrow) - D_test.y(i_true); %#ok<AGROW>
    ex_z(end+1) = hz(xrow) - D_test.x(i_true); ey_z(end+1) = hz(yrow) - D_test.y(i_true); %#ok<AGROW>
end
H.nrmse_forced_x = sqrt(mean(ex_f.^2))/std(D_test.x);
H.nrmse_forced_y = sqrt(mean(ey_f.^2))/std(D_test.y);
H.nrmse_frozen_x = sqrt(mean(ex_z.^2))/std(D_test.x);
H.nrmse_frozen_y = sqrt(mean(ey_z.^2))/std(D_test.y);
H.r = r; H.q = o.q; H.stride = o.stride; H.sv = diag(S(1:min(30,end),1:min(30,end)))';
H.n_params = numel(A) + numel(B);
H.window_s = (o.q-1)*o.stride*dt;
end

function [Hm, j0] = hankel_xy(D, q, s)
% column j holds [x(j0+j-1), x(j0+j-1-s), ..., y(...)]: backward looking
N = numel(D.x); j0 = (q-1)*s + 1; M = N - j0 + 1;
Hm = zeros(2*q, M);
for d = 0:q-1
    Hm(d+1, :)   = D.x(j0 - d*s : j0 - d*s + M - 1)';
    Hm(q+d+1, :) = D.y(j0 - d*s : j0 - d*s + M - 1)';
end
end

function z = rk4_lin(A, B, z, u0, u1, dt)
um = 0.5*(u0+u1);
k1 = A*z + B*u0; k2 = A*(z+dt/2*k1) + B*um; k3 = A*(z+dt/2*k2) + B*um; k4 = A*(z+dt*k3) + B*u1;
z = z + dt/6*(k1+2*k2+2*k3+k4);
end

function E = expm_local(M)
E = expm(M);
end
