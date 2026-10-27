function net = nn_fit(Xi_train, r_train, Xi_val, r_val, varargin)
% NN_FIT  Small fully-connected residual network r ~ f_NN(xi),
%   f_NN(xi) = W_L tanh(... tanh(W_1 xi + b_1) ...) + b_L,
% trained by Adam with early stopping on a validation split; inputs and
% outputs are standardized. Used by spiral_po_loop.m and by the
% residual-predictability check (diagnostics.m, Step 7).
%
%   net = nn_fit(Xi_train, r_train, Xi_val, r_val, 'hidden', [12 12], 'seed', 1)
%   rhat = net.predict(Xi)
%
% Options: 'hidden' ([16 16]), 'lr' (1e-3), 'max_epochs' (2000),
% 'patience' (50), 'batch_size' (256), 'seed' (0)

p = inputParser;
p.addParameter('hidden', [16 16]);
p.addParameter('lr', 1e-3);
p.addParameter('max_epochs', 2000);
p.addParameter('patience', 50);
p.addParameter('batch_size', 256);
p.addParameter('seed', 0);
p.parse(varargin{:});
o = p.Results;

rand('seed', o.seed); randn('seed', o.seed);

[Ntr, d] = size(Xi_train);
k = size(r_train, 2);
layers = [d, o.hidden, k];
L = numel(layers) - 1;

% --- standardize inputs and outputs ---
mu_x = mean(Xi_train,1); sd_x = std(Xi_train,[],1); sd_x(sd_x==0) = 1;
mu_y = mean(r_train,1);  sd_y = std(r_train,[],1);  sd_y(sd_y==0) = 1;
Xs_tr = (Xi_train - mu_x) ./ sd_x;
Ys_tr = (r_train  - mu_y) ./ sd_y;
Xs_val = (Xi_val - mu_x) ./ sd_x;
Ys_val = (r_val  - mu_y) ./ sd_y;

% --- Glorot initialization ---
W = cell(L,1); b = cell(L,1);
for l = 1:L
    fan_in = layers(l); fan_out = layers(l+1);
    s = sqrt(2/(fan_in+fan_out));
    W{l} = s*randn(fan_out, fan_in);
    b{l} = zeros(fan_out,1);
end

% --- Adam state ---
mW = cell(L,1); vW = cell(L,1); mb = cell(L,1); vb = cell(L,1);
for l = 1:L
    mW{l} = zeros(size(W{l})); vW{l} = zeros(size(W{l}));
    mb{l} = zeros(size(b{l})); vb{l} = zeros(size(b{l}));
end
beta1=0.9; beta2=0.999; eps_adam=1e-8;

best_val = inf; best_W = W; best_b = b; bad_epochs = 0;
nstep = 0;

for epoch = 1:o.max_epochs
    idx = randperm(Ntr);
    for bstart = 1:o.batch_size:Ntr
        bidx = idx(bstart:min(bstart+o.batch_size-1, Ntr));
        Xb = Xs_tr(bidx,:)';   % d x B
        Yb = Ys_tr(bidx,:)';   % k x B
        B = size(Xb,2);

        % forward
        A = cell(L+1,1); Zc = cell(L,1);
        A{1} = Xb;
        for l = 1:L
            Zc{l} = W{l}*A{l} + b{l};
            if l < L
                A{l+1} = tanh(Zc{l});
            else
                A{l+1} = Zc{l};   % linear output layer
            end
        end
        Yhat = A{L+1};
        dY = (Yhat - Yb) / B;     % dMSE/dYhat

        % backward
        gW = cell(L,1); gb = cell(L,1);
        delta = dY;
        for l = L:-1:1
            gW{l} = delta * A{l}';
            gb{l} = sum(delta,2);
            if l > 1
                delta = (W{l}' * delta) .* (1 - A{l}.^2);  % tanh'
            end
        end

        % Adam update
        nstep = nstep + 1;
        for l = 1:L
            mW{l} = beta1*mW{l} + (1-beta1)*gW{l};
            vW{l} = beta2*vW{l} + (1-beta2)*gW{l}.^2;
            mb{l} = beta1*mb{l} + (1-beta1)*gb{l};
            vb{l} = beta2*vb{l} + (1-beta2)*gb{l}.^2;
            mWhat = mW{l}/(1-beta1^nstep); vWhat = vW{l}/(1-beta2^nstep);
            mbhat = mb{l}/(1-beta1^nstep); vbhat = vb{l}/(1-beta2^nstep);
            W{l} = W{l} - o.lr * mWhat ./ (sqrt(vWhat)+eps_adam);
            b{l} = b{l} - o.lr * mbhat ./ (sqrt(vbhat)+eps_adam);
        end
    end

    % --- validation check (early stopping) ---
    Yval_hat = forward_predict(W, b, Xs_val');
    val_mse = mean((Yval_hat' - Ys_val).^2, 'all');
    if val_mse < best_val - 1e-9
        best_val = val_mse; best_W = W; best_b = b; bad_epochs = 0;
    else
        bad_epochs = bad_epochs + 1;
        if bad_epochs >= o.patience
            break;
        end
    end
end

net.W = best_W; net.b = best_b;
net.mu_x = mu_x; net.sd_x = sd_x; net.mu_y = mu_y; net.sd_y = sd_y;
net.epochs_run = epoch;
net.best_val_mse = best_val;
net.predict = @(Xi) predict_unstd(net, Xi);

end % function nn_fit


function Yhat = forward_predict(W, b, Xs)
% Xs: d x B (already standardized)
L = numel(W);
A = Xs;
for l = 1:L
    Z = W{l}*A + b{l};
    if l < L
        A = tanh(Z);
    else
        A = Z;
    end
end
Yhat = A;   % k x B
end


function Y = predict_unstd(net, Xi)
Xs = (Xi - net.mu_x) ./ net.sd_x;
Yhats = forward_predict(net.W, net.b, Xs')';
Y = Yhats .* net.sd_y + net.mu_y;
end
