function res = plotLLvsDim(runIdx,varargin)
% res = plotLLvsDim(runIdx,...)
%
% Plot cross-validated log-likelihood versus state dimensionality
% for GPFA and TD-GPFA
%
% INPUTS:
%
% runIdx    - results files will be loaded from mat_results/runXXX, where
%             XXX is runIdx
%
% OPTIONAL ARGUMENTS:
%
% cutoffPC  - cutoff percentage. Used to define the "elbow" of the curve,
%             where the elbow is the
%             lowest latent dimensionality that satisfies
%             cross validated log likelihood > cutoffPC*(height of cvLL curve)
%             (Default = 0.9)
% 
% OUTPUTS:
%
% res       - data structure containing log-likelihood values shown in plot
%
% @ 2015 Karthik Lakshmanan (karthikl@cs.cmu.edu)

plotOn   = true;
cutoffPC = 0.9; 
assignopts(who, varargin);

runDir = sprintf('mat_results/run%03d', runIdx);
myColors = colorspecifier();    
if ~isdir(runDir)
    fprintf('ERROR: %s does not exist.  Exiting...\n', runDir);
    return
else
    D = dir([runDir '/*.mat']);
end


if isempty(D)
    fprintf('ERROR: No valid files.  Exiting...\n');
end
methods = {'gpfa','tdgpfa'};
colors = {myColors.blue,myColors.red};
for i = 1:length(D)
    P = parseFilename(D(i).name);
    D(i).method = P.method;
    D(i).xDim   = P.xDim;
    D(i).cvf    = P.cvf;
    [~, D(i).methodIdx] = ismember(D(i).method, methods);
end

% Only continue processing files that have test trials
D = D([D.cvf]>0);
% Only continue processing files with a method listed in allMethods
D = D([D.methodIdx]>0);

if isempty(D)
    fprintf('ERROR: No valid files.  Exiting...\n');
    return;
end
  
% For each method, find cross validated log-likelihoods for each 
% latent dimensionality

xDims = unique([D.xDim]);
numFolds = length(unique([D.cvf]));
methodIds = unique([D.methodIdx]);
numMethods = length(methodIds);
cvLL = cell(numMethods,1);
for methodIdx = 1:numMethods
    cvLL{methodIdx} = nan(numFolds,length(xDims));     
end

for i = 1:length(D)    
    fprintf('Loading %s/%s...\n', runDir, D(i).name);
    ws = load(sprintf('%s/%s', runDir, D(i).name),'LLtest');
    idx = xDims == D(i).xDim;
    cvLL{D(i).methodIdx}(D(i).cvf,idx) = ws.LLtest;
end 
h = figure;
hold on; 

% Divide by number of folds and plot mean cvLL versus latent dimensionality
legendEntries = {};
for methodIdx = methodIds  
    
    if plotOn
        cv_LL_mean = mean(cvLL{methodIdx},1);
        plot(xDims(~isnan(cv_LL_mean)),cv_LL_mean(~isnan(cv_LL_mean))','o-',...
            'MarkerFaceColor', colors{methodIdx}, ...
            'MarkerEdgeColor', colors{methodIdx}, ...
            'LineWidth', 1,...
            'Color',colors{methodIdx});
        [cv_LL_max,max_id] = max(cv_LL_mean);
        [cv_LL_min,min_id] = min(cv_LL_mean);
        plot(xDims(max_id), cv_LL_mean(max_id),'o',...
            'MarkerSize', 10,...
            'MarkerFaceColor', [1 1 1], ...
            'MarkerEdgeColor', colors{methodIdx}, ...
            'LineWidth', 1,...
            'Color',colors{methodIdx});
        elbow_id = xDims == min(xDims(cv_LL_mean > cv_LL_min + ...
            cutoffPC*(cv_LL_max - cv_LL_min)));
        plot(xDims(elbow_id), cv_LL_mean(elbow_id),'p',...
            'MarkerSize', 10,...
            'MarkerFaceColor', colors{methodIdx}, ...
            'MarkerEdgeColor', colors{methodIdx}, ...
            'LineWidth', 1,...
            'Color',colors{methodIdx});
        legendEntries = [legendEntries {methods{methodIdx}} ...
            {strcat(methods{methodIdx},' peak')}...
            {strcat(methods{methodIdx},' elbow')}];
    end
    
    res{methodIdx}.name = methods{methodIdx};
    res{methodIdx}.cvLL = cv_LL_mean(~isnan(cv_LL_mean));
    res{methodIdx}.xDims = xDims(~isnan(cv_LL_mean));
end
legend(legendEntries,'Location','southeast');
xlabel('Latent dimensionality');
ylabel('Cross validated Log-likelihood');
hold off;

