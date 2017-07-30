function [K_big] = make_K_big_tdgpfa(params, T)
%
% [K_big] = make_K_big(params, T)
%
% Constructs full GP covariance matrix across all state dimensions and
% timesteps.
%
% INPUTS:
%
% params       - GPFA model parameters
% T            - number of timesteps
%
% OUTPUTS:
%
% K_big        - GP covariance matrix with dimensions 
%                (xDim * yDim * T) x (xDim * yDim * T).                
%
% @ 2015 Karthik Lakshmanan karthikl@cs.cmu.edu

yDim         = params.yDim;
xDim         = params.xDim;
K_big        = zeros(xDim*yDim*T);
qT           = yDim*T;

Tdif = repmat(1:T,yDim,1); 
Tdif = repmat(Tdif(:)',qT,1) - repmat(Tdif(:),1,qT);
for i = 1:xDim
    Delayall = params.DelayMatrix(:,i);
    Delaydif = repmat(Delayall,T,1); 
    Delaydif = repmat(Delaydif',qT,1) - repmat(Delaydif,1,qT);
    deltaT = Tdif - Delaydif; 
    deltaTsq = deltaT.^2;
    switch(params.covType)
        case 'rbf'
            temp = exp(-0.5*params.gamma(i)*deltaTsq);          
    end
    K_i = (1-params.eps(i))*temp + params.eps(i)*eye(qT);
    
    idx = i:xDim:xDim*yDim*T;
    K_big(idx,idx) = K_i;
end
