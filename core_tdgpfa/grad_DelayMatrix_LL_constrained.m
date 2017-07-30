function [L,dL] = grad_DelayMatrix_LL_constrained(p,Tall,const)
% [L,dL] = grad_DelayMatrix_LL_constrained(p,seq,const)
%
% Compute gradient wrt log likelihood for ECME
% This function is called by minimize.m.
%
% INPUTS:
%
% p           - variable with respect to which optimization is performed,
%               where p is a vector containing values of the DelayMatrix
%               transformed by a logistic function 
% Tall         - [1 x number of trials] vector of all trial lengths
% const      - structure containing parameters that stay constant for this
%              optimization
%
% OUTPUTS:
%
% f           - value of objective function log P({y})] at p
% df          - gradient at p    
%
% @ 2015 Karthik Lakshmanan    karthikl@cs.cmu.edu

yDim = const.params.yDim;
xDim = const.params.xDim;
gamma = const.params.gamma;
precomp = const.precomp;
Tu = unique(Tall);
L = 0;
dL = zeros(size(p));

for j = 1:length(Tu)
    T = Tu(j);
    qT = yDim*T;
    Tdif = repmat(1:T,yDim,1);
    Tdif = repmat(Tdif(:)',yDim*T,1) - repmat(Tdif(:),1,yDim*T);
    K_big        = zeros(xDim*yDim*T);
    dtemp = zeros(qT,qT,xDim);
    for i = 1:xDim
        Betaall =  [0;p((i-1)*(yDim-1)+1:i*(yDim-1))]; 
        Delayall = 2*const.params.maxDelay./(1+exp(-Betaall)) ...
            - const.params.maxDelay;
        Delaydif = repmat(Delayall,T,1); 
        Delaydif = repmat(Delaydif',qT,1) - repmat(Delaydif,1,qT);
        deltaT = Tdif-Delaydif; 
        deltaTsq = deltaT.^2;
        temp = (1-const.params.eps(i))*exp(-gamma(i)/2 * deltaTsq);
        dtemp(:,:,i) = gamma(i)*temp.*deltaT;
        idx = i:xDim:xDim*yDim*T;
        K_i = temp + const.params.eps(i)*eye(qT); 
        K_big(idx,idx) = K_i;
    end
    C_tilde = precomp.Tu(j).C_tilde;
    R_tilde = precomp.Tu(j).R_tilde;    
    CKC_plus_R_tilde = C_tilde*K_big*C_tilde' + R_tilde;
    try
        CKC_plus_R_tilde_inv = invChol_mex(CKC_plus_R_tilde);
    catch
        CKC_plus_R_tilde_inv = eye(qT)/(CKC_plus_R_tilde);
    end
    nList    = find(Tall == T);
    
    L_term1 = -yDim*T/2*log(2*pi) + (1/2)*logdet(CKC_plus_R_tilde_inv);    
    L = L + length(nList)*L_term1 - 0.5*CKC_plus_R_tilde_inv(:)'*precomp.Tu(j).y_d_SUM(:);
    
    dL_dSigma = (-0.5*length(nList)*eye(qT) + ...
        0.5*CKC_plus_R_tilde_inv*precomp.Tu(j).y_d_SUM)*CKC_plus_R_tilde_inv;
    CdL_dSigmaC = C_tilde'*dL_dSigma*C_tilde;
    dKi_dBetak = zeros(yDim*T);
    
    for i = 1:xDim
        Betaall =  [0;p((i-1)*(yDim-1)+1:i*(yDim-1))]; 
        temp = dtemp(:,:,i);
        idx_big = i:xDim:xDim*yDim*T;
        CdL_dSigmaC_i =CdL_dSigmaC(idx_big,idx_big);

        exp_Beta = exp(-Betaall); 
        dDelayall_dBetaall = -2*const.params.maxDelay*exp_Beta./((1+exp_Beta).^2);       
        
        for k = 2:length(Delayall)            
            idx = k:yDim:yDim*T;
            dKi_dBetak(:,idx) = -temp(:,idx)*dDelayall_dBetaall(k);
            dKi_dBetak(idx,:) = temp(idx,:)*dDelayall_dBetaall(k);
            dKi_dBetak(idx,idx) = 0;   
            dL_dDelay_k = CdL_dSigmaC_i(:)'*dKi_dBetak(:);  
            dKi_dBetak(:,idx) = 0;
            dKi_dBetak(idx,:) = 0;
            dL((i-1)*(yDim-1)+k-1) = dL((i-1)*(yDim-1)+k-1) + dL_dDelay_k;
        end
    end
end

L = -L;
dL = -dL;
