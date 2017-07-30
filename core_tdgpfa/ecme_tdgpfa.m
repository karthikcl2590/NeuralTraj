function [estParams,seq,LL,iterTime] = ecme_tdgpfa(currentParams,seq,varargin)

%
% [estParams, seq, LL] = ecme_tdgpfa(currentParams, seq, ...)
%
% Fits TD-GPFA model parameters using the ECME algorithm.
%
% INPUTS:
%
% currentParams - TD-GPFA model parameters at which EM algorithm is initialized
%                   covType (string)          -- type of GP covariance ('rbf')
%                   gamma (1 x xDim)          -- GP timescales in milliseconds are
%                                               'stepSize ./ sqrt(gamma)'
%                   eps (1 x xDim)            -- GP noise variances
%                   d (yDim x 1)              -- observation mean
%                   C (yDim x (yDim x xDim))  -- mapping between low-d
%                                                and high-d spaces
%                   R (yDim x yDim)           -- observation noise covariance
%                   xDim                      -- latent dimensionality
%                   yDim                      -- number of neurons
% seq           - data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                   trialId      -- unique trial identifier
%                   T (1 x 1)    -- number of timesteps
%                   y (yDim x T) -- neural data
%
% OUTPUTS:
%
% estParams      - learned TD-GPFA model parameters returned by ECME algorithm
%                   (same format as currentParams)
% seq            - data structure with new fields
%                  xsm ((xDim*yDim) x T)           -- posterior mean 
%                                                     at each timepoint
%                  Vsm (xDim*yDim x xDim*yDim x T) -- posterior covariance
%                                                     at each timepoint
%                  VsmGP (yDim*T x yDim*T x xDim)  -- posterior covariance
%                                                     of each GP
% LL             - data log likelihood after each EM iteration
% iterTime       - computation time for each EM iteration
%
% OPTIONAL ARGUMENTS:
%
% ecmeMaxIters   - number of ECME iterations to run (default: 100)
% tol            - stopping criterion for ECME (default: 1e-8) 
% freqLL         - data likelihood is computed every freqLL EM iterations.
%                  freqLL = 1 means that data likelihood is computed every
%                  iteration. (default: 10)
% verbose        - logical that specifies whether to display status messages
%                  (default: false)
%
% @ 2015         Karthik Lakshmanan    karthikl@cs.cmu.edu              

ecmeMaxIters = 100;
tol          = 1e-8;
verbose      = false;
freqLL       = 10;
maxDelayFrac = 0.5;   
parallelize  = false;
extra_opts   = assignopts(who,varargin);

N            = length(seq(:));
T            = [seq.T];
yDim         = size(seq(1).y,1);
xDim         = currentParams.xDim;
LL           = [];
LLi          = -inf;
iterTime     = [];
minVarFrac   = 0.01;
varFloor     = minVarFrac * diag(cov([seq.y]'));

currentParams.maxDelay = round(maxDelayFrac*min([seq.T]));
maxDelay = currentParams.maxDelay;

% constrain all delays to be in (-maxDelay,maxDelay)
currentParams.DelayMatrix(currentParams.DelayMatrix >= maxDelay) = rand;
currentParams.DelayMatrix(currentParams.DelayMatrix <= -maxDelay) = rand;
for i =  1:ecmeMaxIters
    if verbose
        fprintf('\n');
    end
    tic;
    
    if ~parallelize
        fprintf('ECME iteration %3d of %d\n', i, ecmeMaxIters);
    end
    
    if (rem(i, freqLL) == 0) || (i==1) || (i == ecmeMaxIters)
        getLL = true;
    else
        getLL = false;
    end
    
    
    %% === E STEP ===
    
    if ~isnan(LLi)
        LLold = LLi;
    end
    [seq,LLi] = exactInferenceWithLL_tdgpfa(seq, currentParams, 'getLL', getLL);   
    LL = [LL LLi];

    %% === CM STEP 1 ===
    % Learn C,d,R,gamma     
    
    % Solve for C and d together
    C = zeros(yDim,xDim*yDim);
    d = zeros(yDim,1);
    for j = 1:yDim
        sum_A_jj = zeros(xDim+1);
        sum_term = zeros(1,xDim+1);
        idx = 1+(j-1)*xDim:j*xDim;
        for n = 1:N
            for t = 1:seq(n).T
                xsm_j = seq(n).xsm(idx,t);
                sum_Pauto_j = seq(n).Vsm(idx,idx,t);
                sum_A_jj = sum_A_jj + [(sum_Pauto_j + xsm_j*xsm_j') xsm_j; xsm_j' 1];
                y = seq(n).y;
                sum_term = sum_term + y(j,t)*([xsm_j;1])';
            end
        end
        cj_dj = sum_term/sum_A_jj;
        C(j,1+(j-1)*xDim:j*xDim) = cj_dj(1,1:end-1);
        d(j,1) = cj_dj(end);
    end
    currentParams.C = C;
    currentParams.d = d;
    
    % Solve for R
    sum_term1 = 0;sum_term2 = 0;sum_term3 = 0;sum_term4 = 0;sum_term5 = 0;
    for n = 1:N
        for t = 1:seq(n).T
            y = seq(n).y;
            sum_term1 = sum_term1 + y(:,t)*y(:,t)';
            sum_term2 = sum_term2 + (currentParams.d - y(:,t))*seq(n).xsm(:,t)';
            sum_term3 = sum_term3 + seq(n).xsm(:,t)*(currentParams.d - y(:,t))';
            sum_term4 = sum_term4 + y(:,t)*currentParams.d'+currentParams.d*y(:,t)';
            sum_term5 = sum_term5 +seq(n).Vsm(:,:,t)+seq(n).xsm(:,t)*seq(n).xsm(:,t)';
        end
    end
    
    R = (1/sum(T))*(sum_term1 + sum_term2*C' + C*sum_term3 - sum_term4 + C*sum_term5*C' ...
        + sum(T)*currentParams.d*currentParams.d');
    if currentParams.notes.RforceDiagonal
        diag_R = max(varFloor,diag(R));
        currentParams.R = diag(diag_R);
    else
        currentParams.R = (R+R')/2; % ensure symmetry
    end
    
    if currentParams.notes.learnKernelParams
        res = learnGPparams_tdgpfa(seq, currentParams, 'verbose', verbose,...
            'algorithm','ecme',extra_opts{:});
        switch currentParams.covType
            case 'rbf'
                currentParams.gamma = res.gamma;            
        end
               
        if currentParams.notes.learnGPNoise
            currentParams.eps = res.eps;
        end
    end  
    
    %% === CM STEP 2 ===
    % Learn DelayMatrix
    res = learnDelayMatrix_tdgpfa(seq, currentParams, extra_opts{:});
    currentParams.DelayMatrix = res.DelayMatrix;
    
    tEnd    = toc;
    iterTime = [iterTime tEnd];
    
    % Display the most recent likelihood that was evaluated
    if ~parallelize
        if verbose
            if getLL
                fprintf('       lik %f (%.1f sec)\n', LLi, tEnd);
            else
                fprintf('\n');
            end
        else
            if getLL
                fprintf('       lik %f\n', LLi);
            else
                fprintf('\n');
            end
        end
    end
    % Verify that likelihood is growing monotonically
    if i<=2
        LLbase = LLi;
    end
    if (LLi < LLold)
        fprintf('\nError: Data likelihood has decreased from %g to %g\n',...
            LLold, LLi);
        keyboard;
    elseif ((LLi-LLbase) < (1+tol)*(LLold-LLbase))
        break;
    end

end
fprintf('\n');

if ~parallelize
    if length(LL) < ecmeMaxIters
        fprintf('Fitting has converged after %d EM iterations.\n', length(LL));
    end
    
    if any(diag(currentParams.R) == varFloor)
        fprintf('Warning: Private variance floor used for one or more observed dimensions in GPFA.\n');
    end
end
estParams = currentParams;

