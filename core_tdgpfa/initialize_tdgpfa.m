function startParams = initialize_tdgpfa(fname,seqTrain,init_method,varargin)
%
% startParams = initialize_params(fname,seqTrain,init_method,...)
%
% Initialize TD-GPFA parameters for the ECME algorithms
%
% INPUTS:
%
% fname    -   filename where results are saved
% seqTrain -   training data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                 trialId      -- unique trial identifier
%                 T (1 x 1)    -- number of timesteps
%                 y (yDim x T) -- neural data 
% init_method - method used to initialize TD-GPFA. Can be MSISA or GPFA.
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
% OPTIONAL ARGUMENTS:
%
% xDim        -   latent dimensionality (default:3)
%
% @ 2015 Karthik Lakshmanan      karthikl@cs.cmu.edu

xDim          = 3;
parallelize   = false;
extraOpts     = assignopts(who, varargin);

if ~parallelize
    fprintf('Initializing TD-GPFA using %s\n',init_method);
end

switch(init_method)
    case 'MSISA'
        [startParams,descr] = init_MSISA(seqTrain, extraOpts{:},...
            'xDim', xDim, 'parallelize', parallelize);
    case 'GPFA'
        [startParams,descr] = init_GPFA(seqTrain, extraOpts{:}, ...
            'xDim', xDim, 'parallelize', parallelize);        
    otherwise
        fprintf('\nError: Invalid Option\n');
end

% Define parameter constraints
startParams.xDim = xDim;
startParams.yDim = size(seqTrain(1).y,1);
startParams.notes.learnKernelParams = true;
startParams.notes.learnGPNoise      = false;
startParams.notes.RforceDiagonal    = true;
startParams.covType = 'rbf';
save(fname,'descr');
end
