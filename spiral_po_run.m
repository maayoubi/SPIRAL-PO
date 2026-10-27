function spiral_po_run(what)
% SPIRAL_PO_RUN  Run the SPIRAL-PO studies of the paper
% "SPIRAL-PO: Symbolic Identification of Partially Observed Nonlinear
% Dynamics with Application to Rotating Machinery" (Nonlinear Dynamics).
%
% Run from the package folder in MATLAB (R2016b or later) or GNU Octave
% (8.x). No toolboxes are needed.
%
%   spiral_po_run              % everything, in the correct order
%   spiral_po_run('quick')     % short check on the nominal record
%   spiral_po_run('synthetic') % Sect. 4.6-4.11, Tables 3-7   (several hours)
%   spiral_po_run('ekf')       % Sect. 4.12-4.13, Fig. 4, Tables 9-11
%   spiral_po_run('rig')       % Sect. 5, Tables 12-13, Fig. 6
%   spiral_po_run('paths')     % only add the package folders to the path
%
% 'synthetic' runs study_synthetic parts A, B, mc, coef and noise in that
% order (the last three reuse the records cached by A and B). 'ekf' and
% 'rig' are independent. Results are written to results/.
%
% Run times (Octave 8.4, one core): synthetic A and B ~1 h each, mc several
% hours, coef and noise ~15 min, ekf ~15 min, rig ~5 min. See README.md.

if nargin < 1, what = 'all'; end
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'), fullfile(root, 'studies'));
if ~exist(fullfile(root, 'results'), 'dir'), mkdir(fullfile(root, 'results')); end

switch lower(what)
    case 'paths'
        fprintf('SPIRAL-PO folders added to the path.\n');
    case 'quick'
        quick_check();
    case 'synthetic'
        study_synthetic('A'); study_synthetic('B');
        study_synthetic('mc'); study_synthetic('coef'); study_synthetic('noise');
    case 'ekf'
        study_ekf('all');
    case 'rig'
        study_rig('all');
    case 'all'
        spiral_po_run('synthetic'); spiral_po_run('ekf'); spiral_po_run('rig');
    otherwise
        error('spiral_po_run:what', 'unknown option "%s"', what);
end
end

% -------------------------------------------------------------------------
function quick_check()
% Nominal record, weak-form identification.
fprintf('Simulating the nominal benchmark (8 s, 60 -> 5 Hz) ...\n');
D = simulate_rotor(spiral_config('nominal', 31));
for e = {'x','y'}
    W = weak_form(D, e{1});
    fprintf('  %s-equation: selected %s  (|t| > %.2f, %d test functions)\n', ...
        e{1}, strjoin(W.selected, ', '), W.crit, W.n_eq);
end
fprintf('Expected: x: Omega*ydot, (x^2+y^2)*x   y: Omega*xdot, (x^2+y^2)*y\n');
end
