function result = neuralTraj(runIdx, dat, varargin)
%
% result = neuralTraj (runIdx, dat, ...)
%
% Prepare data and extract neural trajectories for a range of latent
% dimensionalities. This function can also handle parallelization of all
% cross-validation folds. Data can be input in one of two formats: as a 0/1
% matrix of spiking activity or a sequence of continuous values (see
% datFormat in OPTIONAL ARGUMENTS)
%
% INPUTS: 
%
% runIdx     -  results files will be saved in mat_results/runXXX,
%               where XXX is runIdx
% dat        -  structure whose nth entry (corresponding to the nth
%               experimental trial) has fields that depend on the value of
%               datFormat (see OPTIONAL ARGUMENTS).
%               if datFormat is 'spikes', nth entry has fields
%                       trialId -- unique trial identifier                       
%                       spikes  -- 0/1 matrix of the raw spiking
%                                  activity across all neurons.  Each
%                                  row corresponds to a neuron. Each
%                                  column corresponds to a 1 msec timestep.
%               if datFormat is 'seq', nth entry has fields
%                       trialId      -- unique trial identifier  
%                       T (1 x 1)    -- number of timesteps
%                       y (yDim x T) -- continuous valued data 
%                                       (Eg: binned spike counts)
%
% OUTPUTS:
%
% result      - structure containing all variables saved in 
%               mat_results/runXXX/ if 'numFolds' is 0.  Else, the
%               structure is empty.
%
% OPTIONAL ARGUMENTS: 
%
% datFormat   - specifies format of input data. This can be either
%               'spikes' or 'seq'. NOTE: The 'seq' option serves two
%               purposes 1) Running a method on spiking activity that has
%               already been binned and square-root transformed. 2) Running
%               a method on continuous valued data (Eg: Simulations 1 and 2
%               in Lakshmanan et al., Neural Computation 2015) 
%               (default: 'spikes')
% method      - method for extracting neural trajectories
%               'gpfa' (default), 'tdgpfa', 'fa', 'ppca', 'pca'
% binWidth    - spike bin width in msec (default: 20)
% numFolds    - number of cross-validation folds (default: 0)
%               0 indicates no cross-validation, i.e. train on all trials.
% xDims       - vector of state dimensionalities (default: [3])
% parallelize - logical that when true, uses Matlab's parfor construct to
%               parallelize each fold and latent dimensionality using
%               multiple cores. Number of workers = numFolds * length of
%               xDims (default: false)
%
% 2015 Karthik Lakshmanan     karthikl@cs.cmu.edu

datFormat            = 'spikes';
method               = 'gpfa';
binWidth             = 20;
numFolds             = 0;
xDims                = [3];
kernSDList           = 20:5:80; % in msec
parallelize          = false;
extraOpts            = assignopts(who, varargin);

fprintf('\n---------------------------------------\n');
if ~isdir('mat_results')
    mkdir('mat_results');
end

% Make a directory for this runIdx if it doesn't already exist
runDir = sprintf('mat_results/run%03d', runIdx);
if isdir(runDir)
    fprintf('Using existing directory %s...\n', runDir);
else
    fprintf('Making directory %s...\n', runDir);
    mkdir(runDir);
end

% If input is in format 1 (raw spikes), obtain binned spike counts
switch(datFormat)
    case 'spikes'
        seq  = getSeq(dat, binWidth, extraOpts{:});
    case 'seq'
        seq  = dat.seq;
end

if isempty(seq)
    fprintf('Error: No valid trials.  Exiting.\n');
    return;
end

N    = length(seq);
fdiv = floor(linspace(1, N+1, numFolds+1));
cvf_list = 0:numFolds;
i = 1;
for xDim = xDims
    for cvf = cvf_list
        % Set cross-validation folds
        testMask = false(1, N);
        if cvf > 0
            testMask(fdiv(cvf):fdiv(cvf+1)-1) = true;
        end
        trainMask = ~testMask;
        
        if cvf == 0
            % If training on all trials, keep original trial ordering
            tr = 1:N;
        else
            % Randomly reorder trials before partitioning into training and
            % test sets
            rand('state', 0);
            tr = randperm(N);
        end
        trainTrialIdx = tr(trainMask);
        testTrialIdx  = tr(testMask);
        seqTrain      = seq(trainTrialIdx);
        seqTest       = seq(testTrialIdx);
        
        % Remove inactive units based on training set
        hasSpikesBool = (mean([seqTrain.y], 2) ~= 0);
        
        for n = 1:length(seqTrain)
            seqTrain(n).y = seqTrain(n).y(hasSpikesBool,:);
        end
        
        for n = 1:length(seqTest)
            seqTest(n).y = seqTest(n).y(hasSpikesBool,:);
        end
        
        % Check if training data covariance is full rank
        yAll = [seqTrain.y];
        yDim  = size(yAll, 1);
        if rank(cov(yAll')) < yDim
            fprintf('ERROR: Observation covariance matrix is rank deficient.\n');
            fprintf('Possible causes: repeated units, not enough observations.\n');
            fprintf('Exiting...\n');
            return
        end
        
        cvf_params(i).seqTrain = seqTrain;
        cvf_params(i).seqTest = seqTest;
        cvf_params(i).hasSpikesBool = hasSpikesBool;
        
        % Specify filename where results will be saved
        fname = sprintf('%s/%s_xDim%02d', runDir, method, xDim);
        if cvf > 0
            fname = sprintf('%s_cv%02d', fname, cvf);
        end
        cvf_params(i).fname = fname;
        cvf_params(i).xDim = xDim;
        cvf_params(i).cvf = cvf;
        i = i+1;
    end
end

if parallelize
    tmp=gcp('nocreate')
    if ((length(cvf_params) > 1) && isempty(tmp))
        % Start a parallel pool using the default profile to define
        % the number of workers. Note: If this takes up too much
        % CPU, the number of workers can be restricted to 'n' by
        % changing line 165 from parpool to parpool(n)
        parpool
    end
    parfor i = 1:length(cvf_params)
        callDimRedEngine(cvf_params(i).fname, cvf_params(i).seqTrain, ...
            cvf_params(i).seqTest, 'method', method, ...
            'xDim', cvf_params(i).xDim, 'cvf', cvf_params(i).cvf, ...
            'kernSDList', kernSDList, 'parallelize', parallelize, ...
            'hasSpikesBool', cvf_params(i).hasSpikesBool, extraOpts{:});                
    end
else
    for i = 1:length(cvf_params)
        if cvf_params(i).cvf == 0
            fprintf('\n===== Training on all data =====\n');
        else
            fprintf('\n===== Cross-validation fold %d of %d =====\n', ...
                cvf_params(i).cvf, numFolds);
        end
        callDimRedEngine(cvf_params(i).fname, cvf_params(i).seqTrain, ...
            cvf_params(i).seqTest, 'method', method, ...
            'xDim', cvf_params(i).xDim, 'cvf', cvf_params(i).cvf, ...
            'kernSDList', kernSDList, 'parallelize', parallelize, ...
            'hasSpikesBool', cvf_params(i).hasSpikesBool, extraOpts{:});        
    end
end

result = [];
if (nargout == 1) && (numFolds == 0) && exist([cvf_params(1).fname '.mat'], 'file')
    result = load(cvf_params(1).fname);
end