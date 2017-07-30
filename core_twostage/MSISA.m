function [A, S_all, T, varexpl]=MSISA(X_all,xDim,opts)
%
% [A, S_all, T]=MSISA(X_all,xDim,opts)
%
% M-SISA: Extension of Shift Independent Subspace Analysis to handle
% multiple experimental trials
%
% INPUTS:
% X_all        - (# of observations) x (trial length) x (# of trials) data
%                matrix
% xDim         - latent dimensionality 
% opts         - structure that contains the following fields
%                maxiter   --   maximum number of iteration (default: 1000)
%                conv_crit --   Convergence Criterion 
%                               (default relative change of costfunction
%                               of 1e-6)
%                dispiter  --   If 1, display result for each iteration
%                A         --   Initial A
%                S         --   Initial S
%                T         --   Initial T
%                ConstA    --   Constant A else estimate 
%                ConstS    --   Constant S
%                ConstT    --   Constant T
%
%
% OUTPUTS:        
% A            - (# of observations) x xDim mixing matrix
% S_all        - xDim x (trial length) x (# of trials) matrix of estimated
%                sources
% T            - (# of observations) x xDim matrix of time delays
%
% varexpl      - variance explained
% 
% Adapted from SICA.m ny Morten Mørup, Kristoffer Hougaard Madsen and
% Technical University of Denmark (March 2007)
%
% @ 2015 Karthik Lakshmanan karthikl@cs.cmu.edu


if(mod(size(X_all,2),2)==1)
    X_all = X_all(:,1:end-1,:); 
end
varexplold=0;
runit=mgetopt(opts,'runit',0);
num_init = mgetopt(opts,'num_init',10);
if ~isfield(opts,'S') && runit~=1
    for k=1:num_init 
        if opts.dispiter
            disp(['Initializing MSISA with ' num2str(k) ' of ' num2str(num_init) ' initial solutions'])
        end
        optsn=opts;
        optsn.dispiter=0;
        optsn.runit=1;
        optsn.maxiter=15;
        [Aq, Sq, Tq, varexpl]=MSISA(X_all,xDim,optsn);
        if varexpl>varexplold
            varexplold=varexpl;
            A=Aq;
            S=Sq;
            T=Tq;
        end
    end
else
    mx=max(abs(X_all(:)));
    A=mgetopt(opts,'A',mx*randn(size(X_all,1),xDim));
    S=mgetopt(opts,'S',mx*randn(xDim,size(X_all,2),size(X_all,3)));
    T=mgetopt(opts,'T',zeros(size(A)));
end

nyT=mgetopt(opts,'nyT',1);
maxiter=mgetopt(opts,'maxiter',1000);
conv_crit=mgetopt(opts,'convcrit',1e-6);
betamax=mgetopt(opts,'beta',0);
beta=0;
gamma=mgetopt(opts,'gamma',0);
N=size(X_all,2); % trial lengths
numTrials = size(X_all,3); % number of trials

SST_all = zeros(1,numTrials);
for n = 1:numTrials
    SST_all(n) = norm(X_all(:,:,n),'fro')^2;
end

constS=mgetopt(opts,'constS',0);
constA=mgetopt(opts,'constA',0);
constT=mgetopt(opts,'constT',0);
dispiter=mgetopt(opts,'dispiter',1);

f=1i*2*pi*[0:N-1]/N;

Xf_all = fft(X_all,[],2);
Sf_all = fft(S,[],2);
Xf_all = Xf_all(:,1:floor(size(Xf_all,2)/2)+1,:);
Sf_all = Sf_all(:,1:floor(size(Sf_all,2)/2)+1,:);
f=f(1:size(Xf_all,2));

% Initial Cost
Rec_all = zeros(size(X_all));
for i=1:size(A,1)
    for n = 1:numTrials
        Sft = Sf_all(:,:,n).*exp(T(i,:)'*f);
        Sft=[Sft conj(Sft(:,end-1:-1:2))];
        S=real(ifft(Sft,[],2));
        Rec_all(i,:,n)=A(i,:)*S;
    end
end
cost_all = zeros(1,numTrials);
for n = 1:numTrials
    cost_all(n)= 0.5*norm(X_all(:,:,n)-Rec_all(:,:,n),'fro')^2;
end
cost = sum(cost_all);
cost_oldt=cost;
dcost=inf;
varexpl_all = zeros(1,numTrials);
for n = 1:numTrials
    varexpl_all(n)= (SST_all(n)-2*cost_all(n))/SST_all(n);
end
varexpl = mean(varexpl_all);
told=cputime;
iter=0;


% Display algorithm progress
if dispiter
    disp([' '])
    disp(['Multitrial Shifted Independent Subspace Analysis'])
    disp(['A ' num2str(xDim) ' component model will be fitted']);
    dheader = sprintf('%12s | %12s | %12s | %12s | %12s | %12s | %12s','Iteration','Expl. var.','Cost func.','Delta costf.',' Time(s)   ');
    dline = sprintf('-------------+--------------+--------------+--------------+--------------+');
    disp(dline);
    disp(dheader);
    disp(dline);
end

while iter<maxiter & (dcost>=cost*conv_crit | mod(iter,10)==0 )
    
    iter=iter+1;
    
    if mod(iter,10)==0 & dispiter
        disp(dline);
        disp(dheader);
        disp(dline);
    end
    
    % Update S
    if ~constS
        for i=1:size(Sf_all,2)
            AexpTf = (A.*exp(T*f(i)));
            for n = 1:numTrials
                Sf_all(:,i,n)=AexpTf\Xf_all(:,i,n); 
            end
        end
    end
    X_concat = reshape(X_all,size(X_all,1),size(X_all,3)*size(X_all,2),1);
    % Update A and calculate cost function
    if ~constA || (~constS && constT)
        for i=1:size(A,1)
            S_concat = [];
            for n = 1:numTrials
                Sft=Sf_all(:,:,n).*exp(T(i,:)'*f);
                Sft=[Sft conj(Sft(:,end-1:-1:2))];
                S=real(ifft(Sft,[],2));
                S_concat = [S_concat S];
            end
            if ~constA
                A(i,:)=X_concat(i,:)*S_concat'*...
                    inv(S_concat*S_concat'); 
            end
            if  (~constS & constT)
                for n = 1:numTrials
                    Sft=Sf_all(:,:,n).*exp(T(i,:)'*f);
                    Sft=[Sft conj(Sft(:,end-1:-1:2))];
                    S=real(ifft(Sft,[],2));
                    Rec_all(i,:,n)=A(i,:)*S;
                end
            end
            
        end
    end
    
    % Update T
    if ~constT        
        P.A=A;
        P.Xf_all=Xf_all;
        P.Sf_all=Sf_all;
        P.N2=size(X_all,2);
        P.numTrials = numTrials;
        P.w=ones(1,length(f));
        P.w(2:end-1)=2;
        P.f=f;
        P.nyT=nyT;
        [T,nyT,cost_all]=update_TmH(T,P);
        cost_all=cost_all/size(S,2); % Use Parseval identity for cost function
    else
        sse_all = zeros(1,numTrials);
        cost_all = zeros(1,numTrials);
        for n = 1:numTrials
            sse_all(n)=norm(X_all(:,:,n)-Rec_all(:,:,n),'fro')^2;
            cost_all(n)=0.5*sse_all(n);
        end
    end
    cost = sum(cost_all);
    
    dcost=cost_oldt-cost;
    cost_oldt=cost;
    
    for n = 1:numTrials
        varexpl_all(n)= (SST_all(n)-2*cost_all(n))/SST_all(n);
    end
    varexpl = mean(varexpl_all); 
    
    if rem(iter,1)==0 && dispiter
        t=cputime;
        tim=t-told;
        told=t;
        disp(sprintf('%12.0f | %12.4f | %12.4f | %12.4f | %12.4f ', ...
            iter, varexpl,cost,dcost,tim));
    end
end

% Calculate S
S_all = zeros(xDim,size(X_all,2),numTrials);
for n = 1:numTrials
    Sf = Sf_all(:,:,n);
    if mod(N,2)==0
        Sft=[Sf conj(Sf(:,end-1:-1:2))];
    else
        Sft=[Sf conj(Sf(:,end:-1:2))];
    end
    S_all(:,:,n)=real(ifft(Sft,[],2));
end
% Normalize solution and align S and T
[A,S_all]=normalizeSolution(A,S_all);
tmean=mean(T);
T=T-repmat(tmean,[size(T,1),1]);
T = bsxfun(@minus,T,T(1,:));
Sf_all=fft(S_all,[],2);
Sf_all=Sf_all(:,1:floor(size(Sf_all,2)/2)+1,:);

for n = 1:numTrials
    Sf=Sf_all(:,:,n).*exp(tmean'*f);
    if mod(N,2)==0
        Sft=[Sf conj(Sf(:,end-1:-1:2))];
    else
        Sft=[Sf conj(Sf(:,end:-1:2))];
    end
    S_all(:,:,n)=real(ifft(Sft,[],2));
end

% -------------------------------------------------------------------------
% Parser for optional arguments
function var = mgetopt(opts, varname, default, varargin)
if isfield(opts, varname)
    var = getfield(opts, varname);
else
    var = default;
end
for narg = 1:2:nargin-4
    cmd = varargin{narg};
    arg = varargin{narg+1};
    switch cmd
        case 'instrset',
            if ~any(strcmp(arg, var))
                fprintf(['Wrong argument %s = ''%s'' - ', ...
                    'Using default : %s = ''%s''\n'], ...
                    varname, var, varname, default);
                var = default;
            end
        otherwise,
            error('Wrong option: %s.', cmd);
    end
end



% -------------------------------------------------------------------------
% Normalizes S with respect to A
function [A,S]=normalizeSolution(A,S)
numTrials = size(S,3);
d = zeros(size(S,1),numTrials);
for n = 1:numTrials
    d(:,n) = sqrt(sum(S(:,:,n).^2,2));
end
d = mean(d,2);
for n = 1:numTrials
    S(:,:,n)=S(:,:,n)./repmat(d,[1,size(S,2)]);
end
A=A.*repmat(d',[size(A,1),1]);

% -------------------------------------------------------------------------
% Function to update T using Newton-Raphson
function [T,nyT,cost_all]=update_TmH(T,P)
nyT=P.nyT;
Sf_all=P.Sf_all;
A=P.A;
N2=P.N2;
Xf_all=P.Xf_all;
f=P.f;
w=P.w;
numTrials = P.numTrials;

Recfd_all=zeros(size(A,1),size(Sf_all,2),size(A,2),numTrials);
Q_all = zeros(size(Recfd_all));
Recf_all = zeros(size(A,1),size(Sf_all,2),numTrials);
for n = 1:numTrials
    for d=1:size(A,2)
        temp = bsxfun(@times,A(:,d),exp(T(:,d)*f));
        Recfd_all(:,:,d,n) = bsxfun(@times,temp,Sf_all(d,:,n)); 
    end
    
    Recf_all(:,:,n) = sum(Recfd_all(:,:,:,n),3);
    Q_all(:,:,:,n) = bsxfun(@times,Recfd_all(:,:,:,n),conj(Xf_all(:,:,n)-Recf_all(:,:,n)));
    
end

Hdiag_sum_all = zeros(size(A));
Hall_sum_all = zeros(size(A,1),size(A,2),size(A,2));
grad_sum_all = zeros(size(A));
for n = 1:numTrials
    Q = Q_all(:,:,:,n);
    Hdiag = 2*squeeze(sum(bsxfun(@times,(w.*(f.^2)),real(Q)),2));
    Hdiag_sum_all = Hdiag_sum_all+Hdiag;
    Hall=zeros(size(A,1),size(A,2),size(A,2));
    for d=1:size(A,2)
        Recfd = Recfd_all(:,:,:,n);
        temp2 = real(bsxfun(@times,Recfd,conj(Recfd(:,:,d))));
        temp3 = bsxfun(@times,(w.*(f.^2)),temp2);
        Hall(:,:,d) = 2*squeeze(sum(temp3,2));
    end
    Hall_sum_all = Hall_sum_all + Hall;
    grad = squeeze(sum(bsxfun(@times,w.*f,conj(Q)-Q),2));
    grad_sum_all = grad_sum_all+grad;
end

for i=1:size(grad_sum_all,1)
    grad_sum_all(i,:)=grad_sum_all(i,:)/(-diag(Hdiag_sum_all(i,:))-squeeze(Hall_sum_all(i,:,:)));
end

grad = grad_sum_all;

ind1=find(w==2); % Areas used twice
ind2=find(w==1); % Areas used once
cost_old_all = zeros(1,numTrials);
for n = 1:numTrials
    cost_old_all(n) = norm(Xf_all(:,ind1,n)-Recf_all(:,ind1,n),'fro')^2;
    cost_old_all(n)=cost_old_all(n)+0.5*norm(Xf_all(:,ind2,n)-Recf_all(:,ind2,n),'fro')^2;
end
cost_old = sum(cost_old_all);
keepgoing=1;
Told=T;
while keepgoing
    T=Told-nyT*grad;
    cost_all = zeros(1,numTrials);
    for d=1:size(A,2)
        temp2 = bsxfun(@times,A(:,d),exp(T(:,d)*f));
        for n = 1:numTrials
            Recfd_all(:,:,d,n) = bsxfun(@times,temp2,Sf_all(d,:,n));
            Recf_all(:,:,n) = sum(Recfd_all(:,:,:,n),3);
            cost_all(n) = norm(Xf_all(:,ind1,n)-Recf_all(:,ind1,n),'fro')^2;
            cost_all(n) = cost_all(n) + 0.5*norm(Xf_all(:,ind2,n) - Recf_all(:,ind2,n),'fro')^2;
        end
    end
    cost = sum(cost_all);
    
    if cost<=cost_old
        keepgoing=0;
        nyT=nyT*1.2;
    else
        keepgoing=1;
        nyT=nyT/2;
    end
end
T=mod(T,N2);
ind=find(T>floor(N2/2));
T(ind)=T(ind)-N2;

