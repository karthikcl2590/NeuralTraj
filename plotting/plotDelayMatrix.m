function plotDelayMatrix(estParams,binWidth)
%
% plotDelayMatrix(estParams,binWidth,...)
%
% Plot estimated Delay Matrix
%
% INPUTS: 
% estParams - structure of estimated model parameters, with fields
%             xDim          : latent dimensionality
%             yDim          : number of neurons
%             DelayMatrix   : (yDim x xDim) matrix of estimated delays
%             maxDelay      : maximum possible delay value
%
% binWidth  - spike bin width used when fitting model
%
% @ 2015 Karthik Lakshmanan (karthikl@cs.cmu.edu)

myColors = colorspecifier();

% convert delays from bins to milliseconds
DelayMatrix_est = binWidth*estParams.DelayMatrix;
yDim = estParams.yDim;
xDim = estParams.xDim;
h = figure;
hold on;

maxDelay = max(DelayMatrix_est(:));
minDelay = min(DelayMatrix_est(:));

for latent_id = 1:xDim
    subplot(xDim,1,latent_id);
    hold on;    
    title(sprintf('Delays to latent variable %d',latent_id));
    xlim([1,yDim+1]); 
    ylim([minDelay-20,maxDelay+20]);
    h_est = bar([1:yDim],DelayMatrix_est(:,latent_id),0.4);    
    set(h_est,'facecolor',myColors.blue,'edgecolor','none');       
    ylabel('delay (ms)');  
    set(gca,'XTick',[1,yDim]);
    xlabel('Neurons');    
    hold off;
end


