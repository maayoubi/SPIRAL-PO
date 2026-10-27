function dz = rotor_rhs(tt, z, P, Omega_fun, theta_fun)
% ROTOR_RHS  Right-hand side of the 4-DOF vertical-rotor model (Sect. 4.2
% of the paper), q = [x; y; alpha; beta]:
%
%   M qdd + (Cb + Omega(t) G) qd + [KL + KNL(q)] q = f_unb(t) [- F_h]
%
%   M   = diag(m, m, Id, Id)
%   G   = [0 0 0 0; 0 0 0 0; 0 0 0 Ip; 0 0 -Ip 0]
%   KL  = sum_i kLi Li' Li,   Cb = sum_i cbi Li' Li
%   KNL = sum_i kNLi ||Li q||^2 Li' Li        (cubic bearing stiffness)
%   f_unb = m e Omega^2 [cos(theta+phi0); sin(theta+phi0); 0; 0]
%
% Li (2x4) maps q to the displacement of bearing i.
%
%   dz = rotor_rhs(t, z, P, Omega_fun, theta_fun)
%
% P is cfg.params of spiral_config.m. Optional fields (used for the
% structural-mismatch study; omit them for the plain model):
%   P.aniso      2x2 [kx/kL, ky/kL] per bearing (row 1 bearing 1, row 2
%                bearing 2): anisotropic linear bearing stiffness
%   P.gyro_scale scales G (0 = gyroscopic knock-out)
%   P.bw         Bouc-Wen element at bearing 1 in x and y, two extra
%                states w = [wx; wy]:
%                  F_h = L1' kh w,
%                  wd  = A ud - beta |ud| |w|^(n-1) w - gamma ud |w|^n,
%                with u = L1 q.
%
% State z = [q; qd] (8x1), or [q; qd; w] (10x1) when P.bw.on is true.
% Omega_fun(t) [rad/s] and theta_fun(t) = int Omega dt [rad] give the
% spin schedule.

q  = z(1:4);
qd = z(5:8);
Om = Omega_fun(tt);
th = theta_fun(tt);

M = diag([P.m, P.m, P.Id, P.Id]);
G = zeros(4,4); G(3,4) = P.Ip; G(4,3) = -P.Ip;
if isfield(P, 'gyro_scale'), G = P.gyro_scale * G; end

if isfield(P, 'aniso')
    K1 = P.kL1 * diag(P.aniso(1,:));
    K2 = P.kL2 * diag(P.aniso(2,:));
    KL = P.L1'*K1*P.L1 + P.L2'*K2*P.L2;
else
    KL = P.kL1*(P.L1'*P.L1) + P.kL2*(P.L2'*P.L2);
end
Cb = P.cb1*(P.L1'*P.L1) + P.cb2*(P.L2'*P.L2);

r1 = P.L1*q; r2 = P.L2*q;
KNL = P.kNL1*(r1'*r1)*(P.L1'*P.L1) + P.kNL2*(r2'*r2)*(P.L2'*P.L2);

f = [P.m*P.e*Om^2*cos(th+P.phi0); P.m*P.e*Om^2*sin(th+P.phi0); 0; 0];

has_bw = isfield(P, 'bw') && P.bw.on;
Fh = zeros(4,1);
if has_bw
    w = z(9:10);
    Fh = P.L1' * (P.bw.kh * w);
end

qdd = M \ ( f - Cb*qd - Om*G*qd - (KL+KNL)*q - Fh );
dz = [qd; qdd];

if has_bw
    ud = P.L1*qd;
    aw = abs(w);
    wd = P.bw.A*ud - P.bw.beta*abs(ud).*aw.^(P.bw.n-1).*w - P.bw.gamma*ud.*aw.^P.bw.n;
    dz = [dz; wd];
end
end
