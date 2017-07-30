function LL = computeLL_tdgpfa(seq,params,precomp)
% LL = computeLL_tdgpfa(seq,params,precomp)
%
% Compute Log-likelihood of data under the TD-GPFA model
%
% INPUTS:
% seq    - training data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                   trialId      -- unique trial identifier
%                   T (1 x 1)    -- number of timesteps
%                   y (yDim x T) -- neural data
% 
% params - TD-GPFA model parameters 
%                   covType (string)          -- type of GP covariance ('rbf')
%                   gamma (1 x xDim)          -- GP timescales in milliseconds are
%                                               'stepSize ./ sqrt(gamma)'
%                   eps (1 x xDim)            -- GP noise variances
%                   d (yDim x 1)              -- observation mean
%                   C (yDim x (yDim x xDim))  -- mapping between low- and high-d spaces
%                   R (yDim x yDim)           -- observation noise covariance
%                   xDim                      -- latent dimensionality
%                   yDim                      -- number of neurons
%
% OUTPUTS:
% LL    - log-likelihood
%
% @2015   Karthik Lakshmanan    karthikl@cs.cmu.edu

Tall = [seq.T];
yDim = params.yDim;
xDim = params.xDim;

fname = 'grad_DelayMatrix_LL_constrained';
init_p = reshape(params.DelayMatrix(2:end,:),(yDim-1)*xDim,1);

if strcmp(fname,'grad_DelayMatrix_LL_constrained')
    % Optimize wrt beta, where Tau = Taumax (1 - exp(-beta))/(1+exp(-beta));
    if ~isfield(params,'maxDelay')
        params.maxDelay = floor(min(Tall)/4);
    end
    init_p = -log(2*params.maxDelay./(init_p + params.maxDelay) - 1);
end
const.params = rmfield(params,'DelayMatrix');

% Precomputations to speed up gradient computation
if nargin < 3
    Tu = unique(Tall);
    for j = 1:length(Tu)
        % Precompute Sum{(y-d)(y-d)'}
        T = Tu(j);
        temp = zeros(yDim*T);
        nList    = find(Tall == T);
        d_tilde = repmat(params.d,T,1);
        for n = nList
            y_tilde = reshape(seq(n).y,yDim*T,1);
            diff = y_tilde-d_tilde;
            temp = temp+diff*diff';
        end
        precomp.Tu(j).y_d_SUM = temp;
        
        % Precompute C_tilde and R_tilde
        blah = cell(1,T);
        [blah{:}] = deal(params.C);
        precomp.Tu(j).C_tilde = sparse(blkdiag(blah{:}));
        [blah{:}] = deal(params.R);
        precomp.Tu(j).R_tilde = (blkdiag(blah{:}));
    end
end
const.precomp = precomp;
[LL,~] = grad_DelayMatrix_LL_constrained(init_p,Tall,const);
LL = -LL;
