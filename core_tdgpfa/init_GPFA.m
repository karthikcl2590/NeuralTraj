function [startParams,descr] = init_GPFA(seqTrain,varargin)
%
% [startParams,descr] = init_GPFA(seqTrain,varargin)
%
% Initialize TD-GPFA parameters using parameters estimated by GPFA, and
% setting all delays to 0
%
% INPUTS:
%
% seq     -       data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                 trialId      -- unique trial identifier
%                 T (1 x 1)    -- number of timesteps
%                 y (yDim x T) -- neural data 
%
% OUTPUTS:
%
% startParams -   TD-GPFA model parameters at which ECME algorithm is
%                 initialized. Contains the fields
%                   covType (string)          -- type of GP covariance ('rbf')
%                   gamma (1 x xDim)          -- GP timescales in
%                                                milliseconds are
%                                               'stepSize ./ sqrt(gamma)'
%                   eps (1 x xDim)            -- GP noise variances
%                   d (yDim x 1)              -- observation mean
%                   C (yDim x (yDim*xDim))    -- mapping between low- and
%                                                high-d spaces
%                   R (yDim x yDim)           -- observation noise covariance 
%                   DelayMatrix (yDim*xDim)    -- delays from latent to
%                                                 observed variables
% descr       -   Short description of initialization method
%
% OPTIONAL ARGUMENTS:
%
% xDim        -   latent dimensionality (default:3)
% binWidth    -   spike bin width in msec (default: 20)
% startTau    -   GP timescale initialization in msec (default: 100)
% startEps    -   GP noise variance initialization (default: 1e-6)
%
% @ 2015 Karthik Lakshmanan  karthikl@cs.cmu.edu 


xDim          = 3;
binWidth      = 20; % in msec
startTau      = 100; % in msec
startEps      = 1e-3;
covType       = 'rbf';
parallelize   = false;
extraOpts     = assignopts(who, varargin);

% GP noise variance
startParams.eps   = startEps * ones(1, xDim);
% Initialize params using Factor Analysis
if ~parallelize
    fprintf('Initializing parameters using Gaussian process factor analysis...\n');
end
yAll             = [seqTrain.y];
startParams.xDim = xDim;
startParams.yDim = size(yAll,1);
[faParams, faLL] = fastfa(yAll, xDim, extraOpts{:});

startParams.d = mean(yAll, 2);
startParams.C = faParams.L;
startParams.R = diag(faParams.Ph);

% Define parameter constraints
startParams.notes.learnKernelParams = true;
startParams.notes.learnGPNoise      = false;
startParams.notes.RforceDiagonal    = true;
startParams.covType = covType;
startParams.gamma = (binWidth / startTau)^2 * ones(1, xDim);
currentParams = startParams;

% =====================
% Fit model parameters
% =====================
if ~parallelize
    fprintf('\nFitting GPFA model...\n');
end
[estParams, ~, ~, ~] = em(currentParams, seqTrain,...
    'emMaxIters', 100, extraOpts{:}, 'parallelize', parallelize);

startParams.d = estParams.d;
startParams.C = restructure_C(estParams.C,startParams.xDim,startParams.yDim);
startParams.R = estParams.R;
startParams.DelayMatrix = zeros(yDim,xDim);
startParams.covType = 'rbf';

descr = 'Initialized using GPFA. DelayMatrix initialized to zeros';
