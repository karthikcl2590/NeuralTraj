function plotTDGPFAlatentsVsTime(seq,estParams,binWidth,varargin)
%
% plotTDGPFAlatentsVsTime(seq,estParams,...)
%
% Plot each TDGPFA latent variable with respect to time
%
% INPUTS:
%
% seq      - data structure whose nth entry (corresponding to
%                       the nth experimental trial) has fields
%             T   --        number of time points
%             xsm --        ((yDim*xDim) x T) latent variables
%
% estParams - structure of estimated model parameters, with fields
%             xDim          -- latent dimensionality
%             yDim          -- number of neurons
%             DelayMatrix   -- (yDim x xDim) matrix of estimated delays
% 
% binWidth  - spike bin width used when fitting model
%
% OPTIONAL ARGUMENTS:
%
% nPlotMax  - maximum number of trials to plot (default: 20)
% redTrials - vector of trialIds whose trajectories are plotted in red
%             (default: [])
% nCols     - number of subplot columns (default: 4)
%
% @ 2015 Karthik Lakshmanan (karthikl@cs.cmu.edu)

nPlotMax = 20;
nCols     = 4;
redTrials = [];
extraOpts = assignopts(who,varargin);

colorspecifier;
% since seq.xsm is not in causal order (see Lakshmanan et al., 
% Neural Computation 2015), sort points in increasing order of time
sorted_seq = sort_latents(seq,estParams);
xDim   = estParams.xDim;
nCols  = min(xDim,nCols);
nRows  = ceil(xDim / nCols);
XAll   = [sorted_seq.xsm];
tAll   = binWidth*[sorted_seq.timepoints];
maxXsm = max(XAll(:));
minXsm = min(XAll(:));
maxT   = max(tAll(:));
minT   = min(tAll(:));

figure;
for i = 1:xDim
    subplot(nRows,nCols,i);
    hold on;
    title(sprintf('Latent variable %d',i));
    for n = 1:min(length(sorted_seq),nPlotMax)       
        if ismember(seq(n).trialId,redTrials)
            col = 'r';
            lw = 3;
        else
            col = 0.2*[1 1 1];
            lw = 0.02;
        end
        plot(binWidth*sorted_seq(n).timepoints(i,:),sorted_seq(n).xsm(i,:), ...
            'color',col,'LineWidth',lw);
    end
    ylim([minXsm,maxXsm]);
    xlim([minT,maxT]);
    xlabel('time (ms)');
    hold off;
end
