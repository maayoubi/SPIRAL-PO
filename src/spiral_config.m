function cfg = spiral_config(case_name, noise_seed)
% SPIRAL_CONFIG  Synthetic 4-DOF rotor benchmark configuration.
%
%   cfg = spiral_config();
%   cfg = spiral_config('nominal',31);
%   D   = simulate_rotor(cfg);
%
% Cases: nominal, no_duffing, no_gyro (knock-out controls), aniso,
% hysteresis, params, runup (structural mismatch).
% Measurement noise is 2% of each noise-free displacement channel's std.

if nargin < 1 || isempty(case_name)
    case_name = 'nominal';
end
if nargin < 2 || isempty(noise_seed)
    noise_seed = 31;
end

cfg.case       = case_name;
cfg.noise_seed = noise_seed;
cfg.dt         = 0.001;       % output sampling interval [s]
cfg.Tend       = 15;          % record duration [s]
cfg.f_start    = 40;          % initial spin frequency [Hz]
cfg.f_end      = 0;           % final spin frequency [Hz]
cfg.noise_frac = 0.02;
cfg.n_sub      = 4;           % RK4 substeps per output interval

% Disk and bearing geometry
P.m = 1.15;                   % disk mass [kg]
R = 0.0735;                   % disk radius [m]
P.Ip = 0.5*P.m*R^2;           % polar inertia [kg*m^2]
P.Id = 0.5*P.Ip;              % diametral inertia [kg*m^2]

Lspan = 1.0;                  % bearing span [m]
a = 0.25;                     % bearing 1 to disk center [m]
lam = a/Lspan;
lamb = 1 - lam;
h = lam*lamb*Lspan;

P.L1 = [lamb, 0,     0,  h;
        0,    lamb, -h,  0];
P.L2 = [lam,  0,     0, -h;
        0,    lam,   h,  0];

% Linear stiffness of each bearing [N/m].
geometryFactor = lamb^2 + lam^2;
kBx = P.m*(2*pi*16.0)^2 / geometryFactor;
kBy = P.m*(2*pi*17.5)^2 / geometryFactor;

P.kL1 = kBx;
P.kL2 = kBx;
P.aniso = [1, kBy/kBx;
           1, kBy/kBx];

P.kxy = 0.125*sqrt(kBx*kBy);
P.kyx = P.kxy;

% Linear damping of each bearing [N*s/m].
P.cBx = 5.0 / geometryFactor;     % 8.0
P.cBy = 3.0 / geometryFactor;     % 4.8
P.cxy = 0.1 / geometryFactor;     % 0.16
P.cyx = P.cxy;

% Mean damping retained for per-bearing scaling in the 'params' case.
P.cb1 = (P.cBx + P.cBy)/2;
P.cb2 = P.cb1;

% Isotropic cubic stiffness at each bearing [N/m^3].
P.kNL1 = 1.0e11 / (lamb^4 + lam^4);
P.kNL2 = P.kNL1;

% Static unbalance U = m*e [kg*m] and initial phase [rad].
P.e    = 1.8e-5 / P.m;
P.phi0 = 0.4;

P.gyro_scale = 1;
P.bw.on    = false;
P.bw.kh    = 0;
P.bw.A     = 1;
P.bw.beta  = 0;
P.bw.gamma = 0;
P.bw.n     = 1;

switch lower(case_name)
    case 'nominal'
        % nominal benchmark

    case 'no_duffing'
        P.kNL1 = 0;
        P.kNL2 = 0;

    case 'no_gyro'
        P.gyro_scale = 0;

    case 'aniso'
        % anisotropic diagonal stiffness; cross-axis terms unchanged
        P.aniso = [1.3, 0.7;
                   1.3, 0.7];

    case 'hysteresis'
        % Two additional Bouc-Wen states at bearing 1.
        P.bw.on    = true;
        P.bw.kh    = 0.5*P.kL1;
        P.bw.A     = 1;
        P.bw.beta  = 25;
        P.bw.gamma = 25;
        P.bw.n     = 1;

    case 'params'
        % parameters perturbed by up to +/-20 %
        factors = [1.12, 0.86, 1.17, 0.91, 1.08, ...
                   0.83, 1.19, 0.88, 1.05];

        P.m    = P.m   * factors(1);
        P.Ip   = P.Ip  * factors(2);
        P.Id   = P.Id  * factors(3);
        P.kL1  = P.kL1 * factors(4);
        P.kL2  = P.kL2 * factors(5);
        P.kNL1 = P.kNL1* factors(6);
        P.kNL2 = P.kNL2* factors(7);
        P.cb1  = P.cb1 * factors(8);
        P.cb2  = P.cb2 * factors(9);

    case 'runup'
        cfg.f_start = 0;
        cfg.f_end   = 40;

    otherwise
        error('spiral_config:badCase', ...
              'Unknown case "%s".', case_name);
end

% Constant angular acceleration of the selected linear spin schedule.
P.dOmega = 2*pi*(cfg.f_end - cfg.f_start)/cfg.Tend;
cfg.params = P;

cfg.spiral.NH        = 20000;
cfg.spiral.max_iter  = 6;
cfg.spiral.nn_hidden = [12, 12];
cfg.spiral.alpha     = 0.05;
end