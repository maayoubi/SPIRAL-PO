function dz = rotor_rhs(tt, z, P, Omega_fun, theta_fun)
% ROTOR_RHS  Right-hand side of the 4-DOF vertical-rotor model (Sect. 4 and
% Appendix A of the paper), state z = [x; y; alpha; beta; qd] (plus the two
% Bouc-Wen states w when P.bw.on).
%
%   M qdd + (Cb + Omega G) qd + (KL + KNL(q)) q = f_unb(t) [- F_h]
%
%   KL = sum_i Li' Kbi Li,  Cb = sum_i Li' Cbi Li  (2x2 bearing matrices
%   with cross-axis terms kxy, kyx, cxy, cyx)
%   KNL = sum_i kNLi ||Li q||^2 Li' Li
%   f_unb = U [-Om^2 cos(psi) - dOm sin(psi); -Om^2 sin(psi) + dOm cos(psi); 0; 0],
%   psi = theta + phi0, U = m e.
%
%   dz = rotor_rhs(t, z, P, Omega_fun, theta_fun)
%
% P is cfg.params of spiral_config.m; Omega_fun(t) [rad/s] and
% theta_fun(t) = int Omega dt [rad] give the spin schedule, and P.dOmega
% its (constant) rate. Without P.dOmega the force is U Om^2 [cos; sin].

q  = z(1:4);
qd = z(5:8);
Om = Omega_fun(tt);
th = theta_fun(tt);

M = diag([P.m, P.m, P.Id, P.Id]);
G = zeros(4,4); G(3,4) = P.Ip; G(4,3) = -P.Ip;
if isfield(P, 'gyro_scale'), G = P.gyro_scale * G; end

aniso = ones(2,2);
if isfield(P,'aniso')
    aniso = P.aniso;
end
Kcross = zeros(2);
if isfield(P,'kxy'), Kcross(1,2) = P.kxy; end
if isfield(P,'kyx'), Kcross(2,1) = P.kyx; end
K1 = P.kL1*diag(aniso(1,:)) + Kcross;
K2 = P.kL2*diag(aniso(2,:)) + Kcross;
KL = P.L1'*K1*P.L1 + P.L2'*K2*P.L2;

if isfield(P,'cBx') && isfield(P,'cBy')
    Cnom = diag([P.cBx, P.cBy]);
    if isfield(P,'cxy'), Cnom(1,2) = P.cxy; end
    if isfield(P,'cyx'), Cnom(2,1) = P.cyx; end
    cMean = (P.cBx + P.cBy)/2;
    if cMean == 0
        if any(Cnom(:)) || P.cb1 ~= 0 || P.cb2 ~= 0
            error('rotor_rhs:damping','Cannot scale damping by a zero mean.');
        end
        C1 = zeros(2);
        C2 = zeros(2);
    else
        C1 = (P.cb1/cMean)*Cnom;
        C2 = (P.cb2/cMean)*Cnom;
    end
else
    C1 = P.cb1*eye(2);
    C2 = P.cb2*eye(2);
end
Cb = P.L1'*C1*P.L1 + P.L2'*C2*P.L2;

r1 = P.L1*q; r2 = P.L2*q;
KNL = P.kNL1*(r1'*r1)*(P.L1'*P.L1) + P.kNL2*(r2'*r2)*(P.L2'*P.L2);

U = P.m*P.e;
psi = th + P.phi0;
if isfield(P,'dOmega')
    Fx = U*(-Om^2*cos(psi) - P.dOmega*sin(psi));
    Fy = U*(-Om^2*sin(psi) + P.dOmega*cos(psi));
else
    Fx = U*Om^2*cos(psi);
    Fy = U*Om^2*sin(psi);
end
f = [Fx; Fy; 0; 0];

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
