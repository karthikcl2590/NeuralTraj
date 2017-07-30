function [startParams,descr] = init_MSISA(seq,varargin)
% [startParams,descr] = init_MSISA(seq,varargin)
%
% Initialize TD-GPFA parameters using M-SISA (Appendix B of Lakshmanan et
% al., J Neural Computation, 2015)
%
% INPUTS:
% seq     -       data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                 trialId      -- unique trial identifier
%                 T (1 x 1)    -- number of timesteps
%                 y (yDim x T) -- neural data
%
% OUTPUTS:
% startParams -   TD-GPFA model parameters at which ECME algorithm is
%                 initialized. Contains the fields
%                   covType (string)          -- type of GP covariance ('rbf')
%                   gamma (1 x xDim)          -- GP timescales in milliseconds are
%                                               'stepSize ./ sqrt(gamma)'
%                   eps (1 x xDim)            -- GP noise variances
%                   d (yDim x 1)              -- observation mean
%                   C (yDim x (yDim*xDim))  -- mapping between low- and high-d spaces
%                   R (yDim x yDim)           -- observation noise covariance
%                   DelayMatrix (yDim*xDim) -- delays from latent to
%                                                observed variables
% descr       -   Short description of initialization method
%
% OPTIONAL ARGUMENTS:
% xDim        -   latent dimensionality (default:3)
% binWidth    -   spike bin width in msec (default: 20)
% startTau    -   GP timescale initialization in msec (default: 100)
% startEps    -   GP noise variance initialization (default: 1e-6)
%
% @ 2015 Karthik Lakshmanan  karthikl@cs.cmu.edu

xDim          = 3;
binWidth      = 20; % in msec
startTau      = 100; % in msec
startEps      = 1e-6;
parallelize   = false;
extraOpts     = assignopts(who, varargin);

rng('default');
rng(0);

yDim = size(seq(1).y,1);

% cut all trials to Tmin
Tmin = min([seq.T]);
yAll = zeros(yDim,Tmin,length(seq));
for n = 1:length(seq)
    % cut trials to Tmin
    seq(n).y = seq(n).y(:,1:Tmin);
end

% Initialize d
startParams.d = mean([seq.y],2);

% subtract mean and smooth data
d = mean([seq.y],2);
for n = 1:length(seq)
    seq(n).y = bsxfun(@minus,seq(n).y,d);
    yAll(:,:,n) = smoother(seq(n).y,40,binWidth);
end

opts.maxiter = 100;
opts.num_init = 10;

if parallelize
    opts.dispiter = 0;
else
    opts.dispiter = 1;
end

[C,S,T,~] = MSISA(yAll,xDim,opts);

startParams.C = restructure_C(C,xDim,yDim);
% Note: Negate T to match TD-GPFA notation, where a positive delay means 
% the neuron lags the latent variable
startParams.DelayMatrix = -T; 

% Initialize R using residuals
for n = 1:length(seq)
    seq(n).S = S(:,:,n);
end
seq_recon = reconstruct_MSISA(C,seq,T);
yAll = [seq.y];
yestAll = [seq_recon.y];
E = yAll(:,1:size(yestAll,2),:) - yestAll;
startParams.R = diag(diag(cov(E')));

% Initialize gamma and eps
startParams.gamma = (binWidth / startTau)^2 * ones(1, xDim);
startParams.eps   = startEps * ones(1, xDim);

descr = 'Initialized using multitrial SISA';

