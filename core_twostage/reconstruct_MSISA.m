function seq_recon = reconstruct_MSISA(A,seq,T)
% Reconstruct observations from MSISA model

% Inputs: 
% A       - yDim x xDim mixing matrix
% seq     - structure whose n'th element corresponding to the n'th trial
%           has:
%           S (yDim x T) - matrix of sources
%
% T       - yDim x xDim matrix of delays
% Outputs: 
% seq_recon  - structure whose n'th element corresponding to the n'th trial
%           has:
%           y (yDim x T) - reconstructed neural activity
% 
% @ 2013 Karthik Lakshmanan  karthikl@cs.cmu.edu

for n = 1:length(seq)
    Sf = fft(seq(n).S,[],2);
    Sf=Sf(:,1:floor(size(Sf,2)/2)+1);
    N=size(seq(n).S,2);
    f=1i*2*pi*[0:N-1]/N;
    f=f(1:size(Sf,2));
    for y=1:size(A,1)
        Sft=Sf.*exp(T(y,:)'*f);
        Sft=[Sft conj(Sft(:,end-1:-1:2))];
        S=real(ifft(Sft,[],2));
        seq_recon(n).y(y,:)=A(y,:)*S;
    end
    seq_recon(n).T = size(seq_recon(n).y,2);
end