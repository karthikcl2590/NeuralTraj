function [seq, LL] = exactInferenceWithLL_tdgpfa(seq, params, varargin)
%
% [seq, LL] = exactInferenceWithLL(seq, params,...)
%
% Extracts latent trajectories given TD-GPFA model parameters.
%
% INPUTS:
%
% seq         - data structure, whose nth entry (corresponding to the nth
%               experimental trial) has fields
%                 y (yDim x T) -- neural data
%                 T (1 x 1)    -- number of timesteps
% params      - TD-GPFA model parameters
%
% OUTPUTS:
%
% seq         - data structure with new fields
%                 xsm ((xDim*yDim) x T)        -- posterior mean at each
%                                                 timepoint
%                 Vsm (xDim*yDim x xDim*yDim x T) -- posterior covariance
%                                                    at each timepoint
%                 VsmGP (yDim*T x yDim*T x xDim)  -- posterior covariance
%                                                    of each GP
% LL          - data log likelihood
%
% OPTIONAL ARGUMENTS:
%
% getLL       - logical that specifies whether to compute data log likelihood
%               (default: false)
%
% @ 2015        Karthik Lakshmanan    karthikl@cs.cmu.edu

getLL = false;
assignopts(who,varargin);
q = params.yDim;
p = params.xDim;
pq = p*q;
C = params.C;
R = params.R;

% Group trials of same length together
Tall = [seq.T];
Tu = unique(Tall);
LL = 0;
for j = 1:length(Tu)
    T = Tu(j);
    
    [K_big] = make_K_big_tdgpfa(params,T);
    K_big = sparse(K_big);
    
    blah = cell(1,T);
    [blah{:}] = deal(C);
    C_tilde = sparse(blkdiag(blah{:}));
    [blah{:}] = deal(R);
    R_tilde = (blkdiag(blah{:}));
    
    KC_t = K_big*C_tilde';       
    KC_times_CKC_plus_R_tilde_inv = KC_t/(C_tilde*KC_t + R_tilde);
    Vsm_big = K_big - KC_times_CKC_plus_R_tilde_inv*C_tilde*K_big;    
    
    % (xDim*yDim) X (xDim*yDim) Posterior covariance for each timepoint
    Vsm = nan(pq,pq,T);
    idx = 1:pq;
    for t = 1:T
        cIdx = pq*(t-1)+idx; %idx(t):idx(t+1)-1;
        Vsm(:,:,t) = Vsm_big(cIdx,cIdx);
    end
    
    % (yDim*T) x (yDim*T) Posterior covariance for each GP
    VsmGP = nan(q*T,q*T,p);
    idx = 0:p:pq*T-p;
    for i = 1:p
        VsmGP(:,:,i) = Vsm_big(idx+i,idx+i);
    end
    
    % Process all trials with length T
    nList    = find(Tall == T);
    d_tilde = repmat(params.d,T,1);
    for n = nList
        % xDim X (yDim * T) Posterior mean
        y_tilde = reshape(seq(n).y,q*T,1);
        diff = y_tilde-d_tilde;
        xsm = KC_times_CKC_plus_R_tilde_inv*diff;
        
        seq(n).xsm = reshape(xsm,p*q,T);
        seq(n).Vsm = Vsm;
        seq(n).VsmGP = VsmGP;
    end
end

if getLL
    LL = computeLL_tdgpfa(seq,params);
else
    LL = NaN;
end
