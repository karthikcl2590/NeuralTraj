function reconstructionError = plotReconstructions(runIdx,xDimTDGPFA,xDimGPFA,varargin)
%
% plotReconstructions(runIdx,xDim_tdgpfa,xDim_tdgpfa,...)
%
% Plot normalized difference in reconstruction error between GPFA and
% TD-GPFA. This produces figures analogous to Figure 3,4 c,d in Lakshmanan
% et al., Neural Computation 2015
%
% INPUTS:
% 
% runIdx       - runIdx for this dataset
% xDimTDGPFA   - latent dimensionality of TD-GPFA model 
% xDimGPFA     - latent dimensionality of GPFA model
%
% OPTIONAL ARGUMENTS:
%
% neuronIds    - indices of example neurons to reconstruct (Figure 3,4 c in
%                Lakshmanan et al., Neural Computation 2015). 
%               (default: [])
% plotOn       - logical that specifies whether or not to display plot
%               (default:true)
%
% @ 2015 Karthik Lakshmanan    karthikl@cs.cmu.edu

neuronIds=[];
plotOn = true;
extraOpts = assignopts(who,varargin);

myColors = colorspecifier();
rmsGPFA = [];
rmsTDGPFA = [];

%% GPFA
D = dir(sprintf('mat_results/run%03d/gpfa_xDim%02d_cv*',runIdx,xDimGPFA));
if isempty(D)
    fprintf(sprintf('Error: Please fit model for GPFA with %d latent dimensions',xDimGPFA));
    return;
end

for i = 1:length(D)
    fname = sprintf('mat_results/run%03d/%s',runIdx,D(i).name);
    gpfa = load(fname);
    cvf = gpfa.cvf;
    % PSTH from data
    psth{cvf} = get_psth(gpfa.seqTest);
    
    % PSTH from GPFA reconstructions
    for n = 1:length(gpfa.seqTest)
        reconGPFA(n).y = gpfa.estParams.C*gpfa.seqTest(n).xsm + ... 
            repmat(gpfa.estParams.d,1,size(gpfa.seqTest(n).xsm,2));
        reconGPFA(n).T = size(gpfa.seqTest(n).xsm,2);
    end
    psthGPFA{cvf} = get_psth(reconGPFA);    
end

%% TDGFPA
D = dir(sprintf('mat_results/run%03d/tdgpfa_xDim%02d_cv*',runIdx,xDimTDGPFA));
if isempty(D)
    fprintf(sprintf('Error: Please fit model for TD-GPFA with %d latent dimensions',xDimTDGPFA));
    return;
end

for i = 1:length(D)    
    fname = sprintf('mat_results/run%03d/%s',runIdx,D(i).name);
    tdgpfa = load(fname);
    cvf = tdgpfa.cvf;
    % PSTH from GPFA reconstructions
    for n = 1:length(tdgpfa.seqTest)
        reconTDGPFA(n).y = tdgpfa.estParams.C*tdgpfa.seqTest(n).xsm + ...
        repmat(tdgpfa.estParams.d,1,size(tdgpfa.seqTest(n).xsm,2));
        reconTDGPFA(n).T = size(tdgpfa.seqTest(n).xsm,2);
    end
    psthTDGPFA{cvf} = get_psth(reconTDGPFA);    
end

%% Find reconstruction errors
for i = 1:length(psth)
    rmsGPFA = [rmsGPFA mean((psthGPFA{cvf} - psth{cvf}).^2,2)];
    rmsTDGPFA = [rmsTDGPFA mean((psthTDGPFA{cvf} - psth{cvf}).^2,2)];
end

% find average CV error
cverrorGPFA = mean(rmsGPFA,2);
cverrorTDGPFA = mean(rmsTDGPFA,2);

% average PSTHs across folds
yDim = size(psth{1},1);
T = size(psth{1},2);
cvPsthGPFA = zeros(yDim,T);
cvPsthTDGPFA = zeros(yDim,T);
cvPSTH = zeros(yDim,T);
for cvf = 1:length(psth)
    cvPsthGPFA = cvPsthGPFA + psthGPFA{cvf};
    cvPsthTDGPFA = cvPsthTDGPFA + psthTDGPFA{cvf};
    cvPSTH = cvPSTH + psth{cvf};
end
cvPsthGPFA = cvPsthGPFA/length(psth);
cvPsthTDGPFA = cvPsthTDGPFA/length(psth);
cvPSTH = cvPSTH/length(psth);

%% Plot histogram of errors
reconstructionError = (cverrorGPFA - cverrorTDGPFA)./(cverrorGPFA);
maxDiff = max(abs(reconstructionError));
binsize = maxDiff/20;
xbins = [-maxDiff:binsize:maxDiff] + binsize/2;

if plotOn
    figure;
    hold on;
    xlim([-maxDiff,maxDiff]);
    hist(reconstructionError,xbins);
    h = findobj(gca,'Type','patch');
    set(h, 'facecolor', myColors.gray,'edgecolor','k');
    ylimit = ylim;
    ylim([ylimit(1),ylimit(2)+2]);
    ylimit = ylim;
    line([0 0],[0,ylimit(2)],...
        'LineStyle','--',...
        'Color','k');
    text(maxDiff/3,ylimit(2)-0.5,'TD-GPFA better');
    text(-maxDiff/2,ylimit(2)-0.5,'GPFA better');
    xlabel('\Delta reconstruction error');
    ylabel('number of neurons');
    hold off;
    
    % Plot reconstructions for example neurons
    if ~isempty(neuronIds)
        figure;
        ymax = max([cvPSTH(:);cvPsthTDGPFA(:);cvPsthGPFA(:)]);
        ymin = min([cvPSTH(:);cvPsthTDGPFA(:);cvPsthGPFA(:)]);
        for i = 1:length(neuronIds)
            n_id = neuronIds(i);
            hSubplot(i) = subplot(1,length(neuronIds),i);
            hold on;
            title(sprintf('$$y^{%d}$$ ',n_id),'interpreter','latex','rot',0);
            h = bar(tdgpfa.binWidth*(1:T),cvPSTH(n_id,:));
            set(h, 'facecolor', myColors.paleblue, ...
                'edgecolor',myColors.paleblue);
            h1 = plot(tdgpfa.binWidth*(1:T),cvPsthTDGPFA(n_id,:),...
                'Color',myColors.red,...
                'LineWidth',1);
            h2 = plot(gpfa.binWidth*(1:T),cvPsthGPFA(n_id,:),...
                'Color',myColors.blue,...
                'LineWidth',1);
            xlabel('Time (ms)');
            xlim([0,tdgpfa.binWidth*T]);
            ylabel('Trial-averaged activity');  
            ylim([ymin,ymax]);
            hold off;
        end
        linkaxes(hSubplot);
        legend([h1 h2],{'TD-GPFA','GPFA'},'Location','northeast');
    end
end
