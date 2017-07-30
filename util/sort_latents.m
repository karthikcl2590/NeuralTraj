function sortedSeq = sort_latents(seq,params)
%
% sortedSeq = sort_latents(seq,params)
%
% Sort points in each latent in causal order (latents as learnt are not
% necessarily in causal order, but in grids sorted by neurons. 
% (See Methods in Lakshmanan et al., J Neural Computation 2015)

% INPUTS:
% seq      -           data structure whose nth entry (corresponding to
%                      the nth experimental trial) has fields
%          T           -- number of time points
%          xsm         -- ((yDim*xDim) x T) latent variables
%
% params  -             parameter structure that contains the fields
%          xDim        -- latent dimensionality
%          yDim        -- number of neurons
%          DelayMatrix -- (yDim x xDim) matrix of estimated delays
%          
% OUTPUTS: 
% sortedSeq -          data structure, whose nth entry (corresponding to
%                      the nth experimental trial) has fields
%          xsm        --  (xDim x (T * yDim)), where the
%                         points are all sorted causally
%          timepoints --  (xDim x (T*yDim)) timepoints  (x-axis) to plot
%                         xsm against
%
% @ 2015 Karthik Lakshmanan (karthikl@cs.cmu.edu)

yDim = params.yDim;
xDim = params.xDim;
Tall = [seq.T];
Tmax = max(Tall);
TsMax = repmat((1:Tmax),yDim,1);
TsMax = reshape(TsMax,yDim*Tmax,1);
sortedSeq = seq;
sortedSeq = rmfield(sortedSeq,'xsm');
for i = 1:xDim
    Delays_i_max = repmat(params.DelayMatrix(:,i),Tmax,1);
    for j = 1:length(seq)
        T = seq(j).T;
        xsm = seq(j).xsm(i:xDim:xDim*yDim,:);
        xsm = reshape(xsm,1,yDim*T);
        latentUnsorted = xsm';        
        Delays_i = Delays_i_max(1:T*yDim,:);
        latentUnsorted(:,2) =  TsMax(1:T*yDim,:) - Delays_i; 
        latentSorted = sortrows(latentUnsorted,2);
        sortedSeq(j).xsm(i,:) = latentSorted(:,1)';
        sortedSeq(j).timepoints(i,:) = latentSorted(:,2)';
    end
end
