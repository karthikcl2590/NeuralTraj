function PSTH = get_psth(seq)
%
% PSTH = get_psth(seq)
%
% Trial-average binned spiking activity and return the PSTH 
%
% INPUTS:
% seq       - data structure, whose nth entry (corresponding to
%               the nth experimental trial) has fields
%                 trialId      -- unique trial identifier
%                 T (1 x 1)    -- number of timesteps in trial
%                 y (yDim x T) -- neural data
%
% OUTPUTS:
% PSTH       - (yDim x max(T)) trial averaged activity
%
% @2015 Karthik Lakshmanan (karthikl@cs.cmu.edu)

Tall = [seq.T];
Tmax = max(Tall);
PSTH = zeros(size(seq(1).y,1),Tmax);
for n = 1:length(seq)    
    PSTH(:,1:Tall(n)) = PSTH(:,1:Tall(n)) + seq(n).y;
end

PSTH = PSTH/length(seq);

