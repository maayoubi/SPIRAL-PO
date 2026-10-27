function [Phi, names, info] = build_library(varargin)
% BUILD_LIBRARY  Physics seed and mechanism-labeled candidate library of
% the rotor (Table 1 of the paper).
%
%   [L, names] = build_library(x, y, xd, yd, Omega, Omegadot, eqn)
%       Candidate columns for the x- ('x') or y-equation ('y'), evaluated
%       at arbitrary points (vectors of equal length).
%
%   [P, names, info] = build_library(D, eqn, what, closure_names)
%       Regression matrices on a data record D (simulate_rotor format):
%         what = 'seed'       the four seed columns only
%         what = 'library'    the eight candidates only
%         what = 'joint'      seed columns followed by the listed closure
%                             terms (closure_names, in the given order;
%                             names that are not candidates are skipped)
%         what = 'full'       seed columns followed by all candidates
%       info.target is the measured acceleration of that equation and
%       info.win its SG window.
%
% Seed (always fitted): unbalance forcing and linear damping/stiffness,
%   x-eq:  Omega^2 cos(theta), Omega^2 sin(theta), -xd, -x,
%   theta = int Omega dt (accumulated spin angle, not Omega*t).
%
% Candidates, x-equation (y-equation by exchanging x and y):
%   Omega*ydot        gyroscopic coupling transmitted by the bearings
%   y                 cross-channel (tilt-mediated) linear coupling
%   (x^2+y^2)*x       isotropic cubic bearing stiffness
%   (x^2+y^2)*xdot    nonlinear damping            } competing forms that
%   x^3, x^2*y        single-axis / mixed cubic    } test the selection
%   Omega^2*x         speed-dependent stiffness    }
%   Omegadot*x        rate-dependent stiffness     }

if isstruct(varargin{1})
    [Phi, names, info] = on_record(varargin{:});
    return;
end
[x, y, xdot, ydot, Omega, Omegadot, eqn] = varargin{1:7};
info = struct();
r2 = x.^2 + y.^2;
if strcmpi(eqn, 'x')
    Phi = [Omega.*ydot, y, r2.*x, r2.*xdot, x.^3, x.^2.*y, Omega.^2.*x, Omegadot.*x];
    names = {'Omega*ydot','y','(x^2+y^2)*x','(x^2+y^2)*xdot','x^3','x^2*y','Omega^2*x','Omegadot*x'};
elseif strcmpi(eqn, 'y')
    Phi = [Omega.*xdot, x, r2.*y, r2.*ydot, y.^3, x.*y.^2, Omega.^2.*y, Omegadot.*y];
    names = {'Omega*xdot','x','(x^2+y^2)*y','(x^2+y^2)*ydot','y^3','x*y^2','Omega^2*y','Omegadot*y'};
else
    error('build_library:badEqn', 'eqn must be ''x'' or ''y''');
end
end

% -------------------------------------------------------------------------
function [P, names, info] = on_record(D, eqn, what, closure_names)
if nargin < 3 || isempty(what), what = 'joint'; end
if nargin < 4, closure_names = {}; end
th = cumtrapz(D.t, D.Omega);
if strcmpi(eqn,'x'), s = D.x; sd = D.xd; info.target = D.xdd; info.win = D.win_x;
else,                s = D.y; sd = D.yd; info.target = D.ydd; info.win = D.win_y; end
S = [D.Omega.^2.*cos(th), D.Omega.^2.*sin(th), -sd, -s];
seed_names = {'unb_cos','unb_sin','-self_dot','-self'};
[L, ln] = build_library(D.x, D.y, D.xd, D.yd, D.Omega, D.Omegadot, eqn);
info.theta = th;
switch lower(what)
    case 'seed',    P = S; names = seed_names;
    case 'library', P = L; names = ln;
    case 'full',    P = [S, L]; names = [seed_names, ln];
    case 'joint'
        P = S; names = seed_names;
        for k = 1:numel(closure_names)
            j = find(strcmp(ln, closure_names{k}));
            if isempty(j), continue; end      % seed-type names (no-seed ablation)
            P = [P, L(:,j)]; names{end+1} = closure_names{k}; %#ok<AGROW>
        end
    otherwise
        error('build_library:what', 'unknown option %s', what);
end
end
