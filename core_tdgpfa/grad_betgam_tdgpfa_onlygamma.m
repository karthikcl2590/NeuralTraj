function [f,df] = grad_betgam_tdgpfa_onlygamma(p,precomp,const)
%
% [f, df] = grad_betgam(p, precomp, const)  
%
% Gradient computation for GP timescale optimization.
% This function is called by minimize.m.
%
% INPUTS:
%
% p           - variable with respect to which optimization is performed,
%               where p = [ log(1 / timescale ^2), Tau(i,2),....Tau(i,q)]'
% precomp     - structure containing precomputations
%
% const       - parameters that stay constant during this optimization
%
% OUTPUTS:
%
% f           - value of objective function E[log P({x},{y})] at p
% df          - gradient at p    
%
% @ 2015 Karthik Lakshmanan    karthikl@cs.cmu.edu

  Tauall = [const.Delaysfixed];
  params = precomp.params;
  df = 0;
  f = 0;
  exp_p = exp(p);
  for j = 1:length(precomp.Tu)
      T = precomp.Tu(j).T;
      qT = params.yDim*T;
      Delaydif = repmat(Tauall,T,1); 
      Delaydif = repmat(Delaydif',qT,1) - repmat(Delaydif,1,qT);
      deltaT = (precomp.Tdif(1:qT,1:qT) - Delaydif); 
      deltaTsq = deltaT.^2;
      temp = (1-const.eps)*exp(-(exp_p/2) * deltaTsq);
      K = temp + const.eps*eye(qT);
      KinvPautoSUM = K\precomp.Tu(j).PautoSUM;
      dE_dK = -0.5*(precomp.Tu(j).numTrials*eye(qT) - KinvPautoSUM)/K;
      dK_dgamma = -0.5*temp.*deltaTsq;
      dE_dgamma = dE_dK(:)' * dK_dgamma(:);
      df(1) = df(1) + dE_dgamma;
      f = f - 0.5*precomp.Tu(j).numTrials*logdet(K) -0.5*trace(KinvPautoSUM); 
  end
f = -f;
df(1) = df(1)*exp_p;
df = -df;
