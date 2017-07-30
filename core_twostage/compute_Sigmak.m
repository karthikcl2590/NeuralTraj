function Sigma_k = compute_Sigmak(Y,k)
% Compute emperical estimate of Sigma_k = E[ (Y_t+k)(Y_t)']

% Inputs:
% Y (yDim X T)   zero mean vector of observations
% k (1 x1)       number of time steps to look forward to compute Sigma_k

T = size(Y,2);
Sigma_k = zeros(size(Y,1));
for t = 1:T-k
    Sigma_k = Sigma_k + Y(:,t+k)*Y(:,t)';    
end
Sigma_k = Sigma_k/(T-k);

