function tdgpfaEngine(seqTrain,seqTest,fname,varargin)
%
% tdgpfaEngine(seqTrain, seqTest, fname, ...)
%
% Extract neural trajectories using GPFA
%
% INPUTS:
%
% seqTrain      - training data structure, whose nth entry (corresponding
%                 to the nth experimental trial) has fields
%                   trialId (1 x 1)   -- unique trial identifier
%                   y (# neurons x T) -- neural data
%                   T (1 x 1)         -- number of timesteps
% seqTest       - test data structure (same format as seqTrain)
% fname         - filename of where results are saved
%
% OPTIONAL ARGUMENTS:
%
% xDim          - state dimensionality (default: 3)
% binWidth      - spike bin width in msec (default: 20)
% startTau      - GP timescale initialization in msec (default: 100)
% startEps      - GP noise variance initialization (default: 1e-3)
%
% @ 2015        Karthik Lakshmanan    karthikl@cs.cmu.edu       
%

xDim          = 3;
binWidth      = 20;  % in msec
startTau      = 100; % in msec
startEps      = 1e-6;
init_method   = 'MSISA';
extraOpts     = assignopts(who, varargin);

% ========================================
% Initialize model parameters
% ========================================
startParams = initialize_tdgpfa(fname, seqTrain, init_method, ...
    'binWidth', binWidth, 'covType', 'rbf', 'xDim', xDim, ...
    'startEps', startEps, 'startTau',startTau, extraOpts{:});
currentParams = startParams;

% =====================
% Fit model parameters
% =====================
[estParams, seqTrain, LL, iterTime] = ecme_tdgpfa(currentParams,...
    seqTrain, 'xDim', xDim, extraOpts{:});

% Extract neural trajectories for original, unsegmented trials
% using learned parameters
[seqTrain, LLorig] = exactInferenceWithLL_tdgpfa(seqTrain, estParams);

% ========================================
% Compute Log likelihood of test data
% ========================================
if ~isempty(seqTest)
    [seqTest, LLtest] = exactInferenceWithLL_tdgpfa(seqTest,...
        estParams, 'getLL', true);
end

% ===========
% Save results
% ===========

% To save disk space, only save posterior means xsm. Posterior covariances
% can be re-calculated using the E-step.
seqTrain = rmfield(seqTrain, 'Vsm');
seqTrain = rmfield(seqTrain, 'VsmGP');
if ~isempty(seqTest)
    seqTest = rmfield(seqTest, 'Vsm');
    seqTest = rmfield(seqTest, 'VsmGP');
end

vars = who;
fprintf('Saving %s...\n',fname);
vars = vars(~ismember(vars,[{'extraOpts'},{'yAll'}]));
save(fname,vars{:},'-append');
