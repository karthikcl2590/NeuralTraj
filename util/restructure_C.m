function [C_big] =  restructure_C(C,xDim,yDim) 
%
% [C_big] =  restructure_C(C,xDim,yDim) 
% Converts the loading matrix into the sparse form used by the TD-GPFA
% observation model
%
% INPUTS:
% C     - (yDim x xDim) mapping from low-d to high-d
% xDim  - latent dimensionality
% yDim  - observed dimensionality
%
% OUTPUTS:
% C_big - (yDim x (yDim*xDim)) mapping from low-d to high-d
%
% @2015 Karthik Lakshmanan    karthikl@cs.cmu.edu

C_big = zeros(yDim,xDim*yDim);
j = 1;
for i = 1:yDim
    C_big(i,j:j+xDim-1) = C(i,:);
    j = j + xDim;
end
