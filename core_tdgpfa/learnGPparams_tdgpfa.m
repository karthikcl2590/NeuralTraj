function [res] = learnGPparams_tdgpfa(seq, params, varargin)
%
% [res] = learnGPparams_tdgpfa(seq, params, ...)
%
% Updates parameters of GP state model given neural trajectories.
% Implements (CM step 1 in the ECME algorithm)
%
% INPUTS:
%
% seq         - data structure containing neural trajectories
% params      - current GP state model parameters, which gives starting point
%               for gradient optimization
%
% OUTPUT:
%
% res         - updated GP state model parameters
%
% OPTIONAL ARGUMENTS:
%
% MAXITERS    - maximum number of line searches (if >0), maximum number
%               of function evaluations (if <0), for minimize.m (default:-8)
% verbose     - logical that specifies whether to display status messages
%               (default: false)
%
% @ 2015 Karthik Lakshmanan         karthikl@cs.cmu.edu

MAXITERS  = -12; % for minimize.m
verbose   = false;
assignopts(who, varargin);

switch params.covType
    case 'rbf'
        % If there's more than one type of parameter, put them in the
        % second row of oldParams.
        oldParams = [params.gamma; params.DelayMatrix];
        fname     = 'grad_betgam_tdgpfa_onlygamma';
    % Optional: Insert other covariance functions here    
end

xDim  = params.xDim;
precomp = makePrecomp_tdgpfa(seq,params);

% Loop once for each state dimension (each GP)
gamma = zeros(1,xDim);
DelayMatrix = params.DelayMatrix; 
for i = 1:xDim
    const = [];
    
    if ~params.notes.learnGPNoise
        const.eps = params.eps(i);
    end
    
    switch fname                
        case 'grad_betgam_tdgpfa_onlygamma'
            const.Delaysfixed = DelayMatrix(:,i);
            init_p = log(oldParams(1,i));        
    end   
    
    % This does the heavy lifting
    [res_p, fX, res_iters] =...
        minimize(init_p, fname, MAXITERS, precomp(i), const);
    
    switch params.covType
        case 'rbf'
            switch fname                
                case 'grad_betgam_tdgpfa_onlygamma'
                    DelayMatrix(:,i) = const.Delaysfixed;
                    gamma(i) = exp(res_p);
            end        
    end    
    
    if verbose
        fprintf('\nConverged p; xDim:%d, p:%s', i, mat2str(res_p, 3));
    end
end

res.DelayMatrix = DelayMatrix;
res.gamma = gamma;
res.res_iters = res_iters;
res.fX = fX;

end

