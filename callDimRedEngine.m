function result = callDimRedEngine(fname,seqTrain,seqTest,varargin)
%
% result = callDimRedEngine(fname,seqTrain,seqTest, ...)
%
% Interfaces with twoStage, GPFA and TD-GPFA Engines to extract neural
% trajectories.
%
% INPUTS:
%
% fname       - filename where results will be saved
% seqTrain    - training data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                   trialId      -- unique trial identifier
%                   T (1 x 1)    -- number of timesteps
%                   y (yDim x T) -- neural data
%
% seqTest     - testing data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                   trialId      -- unique trial identifier
%                   T (1 x 1)    -- number of timesteps
%                   y (yDim x T) -- neural data
%
% OUTPUTS:
%
% result      - structure containing all variables saved in
%               mat_results/runXXX/
%               if 'numFolds' is 0.  Else, the structure is empty.
%
% OPTIONAL ARGUMENTS:
%
% method      - method for extracting neural trajectories
%               'gpfa' (default), 'tdgpfa', 'fa', 'ppca', 'pca'
% binWidth    - spike bin width in msec (default: 20)
% numFolds    - number of cross-validation folds (default: 0)
%               0 indicates no cross-validation, i.e. train on all trials.
% xDim        - state dimensionality (default: 3)
%
% @ 2015 Karthik Lakshmanan  karthikl@cs.cmu.edu

method        = 'gpfa';
binWidth      = 20; % in msec
cvf           = 0;
xDim          = 3;
hasSpikesBool = [];
extraOpts     = assignopts(who, varargin);

% If doing cross-validation, don't use private noise variance floor.
if cvf > 0
    extraOpts = {extraOpts{:}, 'minVarFrac', -Inf};
end
extraOpts = {extraOpts{:},'cvf',cvf,'xDim',xDim};

if exist([fname '.mat'], 'file')
    fprintf('%s already exists.  Skipping...\n', fname);
    return;
end

fprintf('Number of training trials: %d\n', length(seqTrain));
fprintf('Number of test trials: %d\n', length(seqTest));
fprintf('Latent space dimensionality: %d\n', xDim);
fprintf('Observation dimensionality: %d\n', size(seqTrain(1).y,1));

% The following does the heavy lifting.
if isequal(method, 'tdgpfa')
    tdgpfaEngine(seqTrain, seqTest, fname,...
        'xDim', xDim, 'binWidth', binWidth, extraOpts{:});
elseif isequal(method,'gpfa')
    gpfaEngine(seqTrain, seqTest, fname,...
        'xDim', xDim, 'binWidth', binWidth, extraOpts{:});
elseif ismember(method, {'fa', 'ppca', 'pca'})
    twoStageEngine(seqTrain, seqTest, fname,...
        'typ', method, 'xDim', xDim, 'binWidth', binWidth, extraOpts{:});
end

if exist([fname '.mat'], 'file')
    save(fname, 'method', 'cvf', 'hasSpikesBool', '-append');
end

result = [];
if (nargout == 1) && (numFolds == 0) && exist([fname '.mat'], 'file')
    result = load(fname);
end