function res = learnDelayMatrix_tdgpfa(seq,params,varargin)
%
% res = learnDelayMatrix_tdgpfa(seq,params, ...)
%
% Updates delays given neural trajectories.
% Implements (CM step 2 in the ECME algorithm)
%
% INPUTS:
%
% seq         - data structure containing neural trajectories
% params      - current GP state model parameters, which gives starting 
%               point for gradient optimization
%
% OUTPUT:
%
% res         - updated delays
%
% OPTIONAL ARGUMENTS:
%
% MAXITERS    - maximum number of line searches (if >0), maximum number
%               of function evaluations (if <0), for minimize.m (default: 30)
% verbose     - logical that specifies whether to display status messages
%               (default: false)
%
% @ 2015 Karthik Lakshmanan         karthikl@cs.cmu.edu

MAXITERS  = 10;
assignopts(who, varargin);

fname = 'grad_DelayMatrix_LL_constrained';

xDim        = params.xDim;
yDim        = params.yDim;
const.params = rmfield(params,'DelayMatrix');
init_p = reshape(params.DelayMatrix(2:end,:),(yDim-1)*xDim,1);

if strcmp(fname,'grad_DelayMatrix_LL_constrained')
    % Optimize wrt beta, where Delay = maxDelay (1 -
    % exp(-beta))/(1+exp(-beta));
    init_p = -log(2*params.maxDelay./(init_p + params.maxDelay) - 1);
end

% Precomputations to speed up gradient computation
Tall = [seq.T];
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
const.precomp = precomp;


[res_p, fX, res_iters] =...
    minimize(init_p, fname, MAXITERS, Tall,const);
    
res_p = reshape(res_p,yDim-1,xDim);
if strcmp(fname,'grad_DelayMatrix_LL_constrained')
    res_p = 2*params.maxDelay./(1+exp(-res_p)) - params.maxDelay;
end
res.DelayMatrix = [zeros(1,xDim); res_p];
res.res_iters = res_iters;
res.fX = fX;

