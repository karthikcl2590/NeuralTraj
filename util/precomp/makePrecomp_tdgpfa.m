function precomp = makePrecomp_tdgpfa(seq,params)
%
% [precomp] = makePrecomp_tdgpfa(seq,params)
%
% Precompute posterior covariances for TD-GPFA algorithm.
%
% INPUTS :
% seq           - data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                   trialId      -- unique trial identifier
%                   T (1 x 1)    -- number of timesteps
%                   y (yDim x T) -- neural data
% params        - parameter information
%
% OUTPUTS :
% precomp - The precomp struct will be updated with the posterior covariance and the other requirements.
% 
% NOTE: It might be a good idea to provide a mex implementation of this.
% @ 2015 Karthik Lakshmanan 

xDim = params.xDim;
yDim = params.yDim;
Tall = [seq.T];
Tmax = max(Tall);
Tdif = repmat(1:Tmax,yDim,1);
Tdif = repmat(Tdif(:)',yDim*Tmax,1) - repmat(Tdif(:),1,yDim*Tmax);
% assign some helpful precomp items
precomp(xDim).Tdif = Tdif;
for i = 1:xDim
    precomp(i).Tdif = Tdif;
    %precomp(i).absDif = abs(Tdif);
    precomp(i).Tall   = Tall;
    precomp(i).params = params;
    
end
% find unique numbers of trial lengths
Tu = unique(Tall);
% Loop once for each state dimension (each GP)
for i = 1:xDim
    for j = 1:length(Tu)
        T     = Tu(j);
        precomp(i).Tu(j).nList = find(Tall == T);
        precomp(i).Tu(j).T = T;
        precomp(i).Tu(j).numTrials = length(precomp(i).Tu(j).nList);
        precomp(i).Tu(j).PautoSUM  = zeros(T*yDim);
    end
end

% Fill out PautoSum
% Loop once for each state dimension (each GP)
for i = 1:xDim
    % Loop once for each trial length (each of Tu)
    for j = 1:length(Tu)
        % Loop once for each trial (each of nList)
        for n = precomp(i).Tu(j).nList
            xsm_i = seq(n).xsm(i:xDim:xDim*yDim,:);
            xsm_i = reshape(xsm_i,1,yDim*Tu(j));
            
            precomp(i).Tu(j).PautoSUM = precomp(i).Tu(j).PautoSUM +...
                seq(n).VsmGP(:,:,i) +...
                xsm_i' * xsm_i;
        end
    end
end




