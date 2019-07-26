function main_MNLMF(cv_flag,gpu_flag) 
%%% cv_flag:  cv_flag=1 represent CV on diseases
%cv_flag=2 represent CV on metabolites
%cv_flag=3 represent CV on interaction
%%%gpu_flag: gpu_flag = 0  Do not use the GPU(Maybe a little slow)
%gpu_flag = 1  Use the GPU (Will be fast)



D_M = chuli_opt();  %Import training data
cv = 5; %5-fold cross validation
seeds = [7771,4659,22,8367,1812]; 
if cv_flag==1  %CVd 
    % 70.0000    2.0000    1.0000   50.0000    0.1000    --- 0.8281
    option = [70.0000    2.0000    1.0000   50.0000    0.1000];
    cv_seed(D_M,seeds,cv_flag,cv,option,gpu_flag)
elseif cv_flag==2 %CVm
    %  50.0000    0.2500    2.0000   50.0000    0.3000  ---  0.9542
    option = [50.0000    0.2500    2.0000   50.0000    0.3000];
    cv_seed(D_M,seeds,cv_flag,cv,option,gpu_flag) 
elseif cv_flag==3 %CVa
    %   50.0000    0.2500    4.0000   50.0000    0.1000  --- 0.9726
    option = [50.0000    0.2500    4.0000   50.0000    0.1000];
    cv_seed(D_M,seeds,cv_flag,cv,option,gpu_flag)   
else
    disp('Input error!');
end


end

%%%%%%%Import training data
function D_M = chuli_opt()
load hmdb_v4.mat;
dis_name = hmdb_v4.dis_name(:,1);
met_name = hmdb_v4.met_name(:,1);
inter_name = hmdb_v4.interaction(:,[1,4]);
[~,indd] = cellfun(@(x)ismember(x,dis_name),inter_name(:,1));    %疾病名字的位置 
[~,indm] = cellfun(@(x)ismember(x,met_name),inter_name(:,2));    %代谢id的位置 
D_M.interaction = full(sparse(indd,indm,ones(length(indd),1)));
D_M.dis_sim = hmdb_v4.dis_sim;
D_M.met_sim = hmdb_v4.met_sim;

end

function result = cv_seed(D_M,seeds,cv_flag,cv,option,gpu_flag)
Ns = length(seeds); 
% num_factors = option(1);  lambda_d = option(3);
% lambda_t = option(2); alpha = option(3);  beta = option(3);
% K = option(4);  num = option(5);  
result = 0;
for i=1:Ns 
    cv_data = cross_validation(D_M.interaction,seeds(i),cv_flag,cv);     
    result0 = fivecross(D_M,cv_data,seeds(i),option,gpu_flag);
    result = result+result0;
end
result = result/Ns;
end

function result = fivecross(D_M,cv_data,seed,option,gpu_flag)
cv = length(cv_data); 
result = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:cv
    k
    scores = MNLMF_opt(D_M,cv_data{k},seed,option,gpu_flag);
    %%%%%%%%%%%%%%%    
    result0 = model_evaluate(scores,cv_data{k}{3}); 
    result = result + result0; 
end
result = result/cv;
end


%reference fromLiu Y, Wu M, Miao C, Zhao P, Li X. Neighborhood Regularized Logistic Matrix Factorization 
% for Drug-Target Interaction Prediction. Plos Comput Biol.
function cv_data = cross_validation(intMat,seeds,cv_flag,cv)
Ns = length(seeds);
cv_data = cell(Ns,cv);  
for i= 1:Ns
    [num_dis,num_miRNA] = size(intMat);   
    ndm = num_dis*num_miRNA; 
    rand('state',seeds(i))
    if cv_flag<=2
        if cv_flag == 1
            index = randperm(num_dis)';
        elseif cv_flag == 2
            index = randperm(num_miRNA)';
        end
        step = floor(length(index)/cv);   
        for j = 1:cv   
            if j < cv
                ii = index((j-1)*step+1:j*step);    
            else
                ii = index((j-1)*step+1:end);
            end
            if cv_flag == 1
                test_data=[];
                for k=1:length(ii)   
                    test_data = [test_data;[ii(k)*ones(num_miRNA,1),[1:num_miRNA]']];
                end
            elseif cv_flag == 2
                test_data=[];
                for k=1:length(ii)
                    test_data = [test_data;[[1:num_dis]',ii(k)*ones(num_dis,1)]];
                end
            end
            test_label = [];
            W = ones(size(intMat));
            for s=1:size(test_data,1)
                test_label = [test_label;intMat(test_data(s,1),test_data(s,2))];
                W(test_data(s,1),test_data(s,2)) = 0;
            end
            cv_data{i,j} = {W, test_data, test_label};
        end
    elseif cv_flag == 3
        ind1 = find(intMat==1); ind0 = find(intMat==0);
        ndg1 = length(ind1);   ndg0 = ndm - ndg1;
        index1 = ind1(randperm(ndg1));
        rand('state',seeds(i))
        index0 = ind0(randperm(ndg0));
        step1 = floor(length(index1)/cv);   
        step0 = floor(length(index0)/cv);   
        for j = 1:cv  
            if j < cv
                ii1 = index1((j-1)*step1+1:j*step1);
                ii0 = index0((j-1)*step0+1:j*step0);
            else
                ii1 = index1((j-1)*step1+1:end);
                ii0 = index0((j-1)*step0+1:end);
            end
            yy1 = ceil(ii1/num_dis);
            xx1 = mod(ii1,num_dis);        xx1(find(xx1==0)) = num_dis;
            yy0 = ceil(ii0/num_dis);       
            xx0 = mod(ii0,num_dis);        xx0(find(xx0==0)) = num_dis;
            test_data = [[xx0,yy0];[xx1,yy1]];  
            test_label = [];
            W = ones(size(intMat));
            for s=1:size(test_data,1)
                test_label = [test_label;intMat(test_data(s,1),test_data(s,2))];
                W(test_data(s,1),test_data(s,2)) = 0;
            end
            cv_data{i,j} = {W, test_data, test_label};
        end
    end
end
end

%%%
function result = model_evaluate(predict_score,real_label) %% evaulate our prediction 

predict_score = (predict_score-min(predict_score))/(max(predict_score)-min(predict_score)); 
threshold = (1:999)/1000; 
predict_matrix = bsxfun(@gt,predict_score(:),threshold);  
TP = sum(bsxfun(@and,predict_matrix,real_label(:)));
FP = sum(bsxfun(@and,predict_matrix,~real_label(:)));
FN = sum(bsxfun(@and,~predict_matrix,real_label(:)));
TN = sum(bsxfun(@and,~predict_matrix,~real_label(:)));
%计算AUC
AUC_x = FP./(TN+FP);      %FPR
AUC_y = TP./(TP+FN);      %TPR 
[AUC_x,ind] = sort(AUC_x);
AUC_y = AUC_y(ind);
AUC_x = [0,AUC_x];
AUC_y = [0,AUC_y];
AUC_x = [AUC_x,1];
AUC_y = [AUC_y,1];
AUC = 0.5*AUC_x(1)*AUC_y(1)+sum((AUC_x(2:end)-AUC_x(1:end-1)).*(AUC_y(2:end)+AUC_y(1:end-1))/2);
% plot(AUC_x,AUC_y)
result = AUC;

end

function scores = MNLMF_opt(D_M,cv_data,seed,option,gpu_flag)
interaction = D_M.interaction;
dis_sim = D_M.dis_sim;
met_sim = D_M.met_sim;
[nd,nm] = size(interaction);
    train_set = interaction.*cv_data{1};    
    Y = xiuzheng_Y(train_set,D_M.dis_sim,D_M.met_sim,10,0.9);   
    %%%%%%disease similarity
    neighbor_num = floor(option(5)*nd);  
    KSNS_SdC = KSNS_opt(Y,neighbor_num,dis_sim,gpu_flag);    
    dis_sim = DCA_opt({KSNS_SdC,D_M.dis_sim},option(4),gpu_flag);      
    %%%%%%metobalite   similarity
    neighbor_num = floor(option(5)*nm);  
    KSNS_SmC = KSNS_opt(Y',neighbor_num,met_sim,gpu_flag);    
    met_sim = DCA_opt({KSNS_SmC,D_M.met_sim},option(4),gpu_flag);     
    %%%%%% Calculated prediction score
    scores = NRLMF_score(Y,dis_sim,met_sim,cv_data{2},seed,option);
end


%%%%Matrix completion  
%reference from Xiao Q, Luo J, Liang C, Cai J, Ding P.
%A graph regularized non-negative matrix factorization method for identifying microRNA-disease associations.
%Bioinformatics. 2018)
function Y = xiuzheng_Y(Y0,dis_sim,dis_met,K,ar)  
ind_dis0 = find(sum(Y0')==0);  
ind_met0 = find(sum(Y0)==0);    
[Nd,Nm] = size(Y0);

%%
dis_sim(1:(Nd+1):end) = 0;     
dis_met(1:(Nm+1):end) = 0; 
dis_sim(ind_dis0,ind_dis0) = 0; 
dis_met(ind_met0,ind_met0) = 0;

[~,indd] = sort(dis_sim,2,'descend');
[~,indm] = sort(dis_met,1,'descend');
Yd = zeros(Nd,Nm);
ar = ar.^(0:K-1);
for i=1:Nd    
    near_ind = indd(i,1:K);
    Yd(i,:) = (ar.*dis_sim(i,near_ind))*Y0(near_ind,:)/sum(dis_sim(i,near_ind));
end
Ym = zeros(Nd,Nm);
for j=1:Nm    
    near_inm = indm(1:K,j);
    Ym(:,j) = Y0(:,near_inm)*(ar'.*dis_met(near_inm,j))/sum(dis_met(near_inm,j));
end 

Y = (Yd+Ym)/2;
Y = max(Y,Y0);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   KSNS
%%reference from Ma Y, Yu L, He T, Hu X, Jiang X (2018) 
%Prediction of Long Non-coding RNA-protein Interaction through Kernel 
%Soft-neighborhood Similarity. 2018 IEEE International Conference on Bioinformatics and Biomedicine 
%(BIBM). 
function S = KSNS_opt(X,neighbor_num,sim,gpu_flag)
% neighbor_num = single(neighbor_num);
if gpu_flag==1
    X = gpuArray(single(X));
    sim = gpuArray(single(sim));
end
    
feature_matrix = X;
nearst_neighbor_matrix = KSNS_neighbors(sim,neighbor_num,gpu_flag);
S = jisuanW_opt(feature_matrix,nearst_neighbor_matrix,gpu_flag);
if gpu_flag==1
    S =  double(gather(S));
end

end

function nearst_neighbor_matrix=KSNS_neighbors(sim,neighbor_num,gpu_flag)
%%%%nearst_neighbor_matrix：  represent Neighbor matrix
  N = size(sim,1);
  D = sim+diag(inf*ones(1,N));
  [~, si]=sort(D,2,'ascend');
  if gpu_flag==1
      nearst_neighbor_matrix=gpuArray.zeros(N,N);
  else
      nearst_neighbor_matrix = zeros(N,N);
  end 
  index=si(:,1:neighbor_num);
  for i=1:N
      nearst_neighbor_matrix(i,index(i,:))=1;     
  end
end

%The iterative process of this algorithm
function [W,objv] = jisuanW_opt(feature_matrix,nearst_neighbor_matrix,gpu_flag)
lata1 = 1;  lata2 = 1;
X=feature_matrix';  % each column of X represents a sample, and each behavior is an indicator
[~,N] = size(X);    % N represents the number of samples
C = nearst_neighbor_matrix';
rand('state',1);
W = rand(N,N);
if gpu_flag == 1
    W = single(W);
end
W = W- diag(diag(W));
W = W./repmat(sum(W),N,1);
G  = jisuan_Kel(X);
G(isnan(G))=0;
G = G/max(G(:));
WC1 = W'*G*W-2*W*G+G;
WC = sum(diag(WC1))/2;
wucha = WC + norm(W.*(1-C),'fro')^2*lata1/2 +  norm(W,'fro')^2*lata2/2;
objv = wucha;
jingdu = 0.0001;
error = jingdu*(1+lata1+lata2);   %Iteration error threshold
we = 1;      %Initial value of error
gen=1;
while  gen<100 && we>error
    %gen
    FZ = G+lata1*C.*W;
    FM = G*W+lata1*W+lata2*W;    
    W = FZ./(FM+eps).*W;  
    WC1 = W'*G*W-2*W*G+G;
    WC = sum(diag(WC1))/2;
    wucha = WC + norm(W.*(1-C),'fro')^2*lata1/2 +  norm(W,'fro')^2*lata2/2;    
    we = abs(wucha-objv(end));
    objv = [objv,wucha];
    gen = gen+1;
end
W=W'; 
W = matrix_normalize(W,gpu_flag);
end

function W = matrix_normalize(W,gpu_flag)
K = 10;
W(isnan(W))=0;
W(1:(size(W,1)+1):end)=0;

if gpu_flag==1
    GW = gpuArray(single(W));
else
    GW = W;
end
for round=1:K
    SW = sum(GW,2);
    ind = find(SW>0);
    SW(ind) = 1./sqrt(SW(ind));
    D1 = diag(SW);
    GW=D1*GW*D1;
end
if gpu_flag==1
    W = gather(GW);
else
    W = GW;
end

end

function K  =jisuan_Kel(X)
%X Columns represent samples, and rows represent features
X = X';
sA = (sum(X.^2, 2));
sB = (sum(X.^2, 2));
K = exp(bsxfun(@minus,bsxfun(@minus,2*X*X', sA), sB')/mean(sA));
end




%%%%%%reference from Wang S, Cho H, Zhai C, Berger B, Peng J. 
%Exploiting ontology graph for predicting sparsely annotated 
%gene function. Bioinformatics.
function S_S = DCA_opt(S,d,gpu_flag)
Ns = length(S);               
option.maxiter =20;
option.reset_prob = 0.5;     
for i=1:Ns
    A = S{i};
    tA = run_diffusion(A, option,gpu_flag);
    if i==1
        QA = tA;
    else
        QA = [QA,tA];
    end    
end

alpha = 1/size(QA,1);
QA = log(QA+alpha)-log(alpha);
QA = QA*QA';

QA = sparse(QA);
[U,S] = svds(QA,d);
LA = U;
S_F = LA*sqrt(sqrt(S));
S_S = abs(cos_opt(S_F));

end

function W = cos_opt(X)
W = X*X';
DX = sqrt(diag(W))*sqrt(diag(W))';
W = W./DX;
end

function [Q] = run_diffusion(A, option,gpu_flag)
n = size(A, 1); 
if gpu_flag==1
    A = gpuArray(single(A));
    reset = gpuArray.eye(n);  
else
    reset = eye(n);
end
    
    
    renorm = @(M) bsxfun(@rdivide, M, sum(M));  
    A = A + diag(sum(A) == 0); 
    P = renorm(A); 
      
    Q = reset;
    for i = 1:option.maxiter 
        Q_new = option.reset_prob * reset + (1 - option.reset_prob) * P * Q;
        delta = norm(Q - Q_new, 'fro');
        Q = Q_new;
        if delta < 1e-6
            break
        end
    end    
    Q = bsxfun(@rdivide, Q, sum(Q));  
    if gpu_flag==1
        Q = double(gather(Q));
    end
end







%%%%%NRLMF
%reference from Liu Y, Wu M, Miao C, Zhao P, Li X. 
%Neighborhood Regularized Logistic Matrix Factorization 
%for Drug-Target Interaction Prediction. Plos Comput Biol. 2016
function scores = NRLMF_score(train_set,dis_sim,met_sim,test_data,seed,canshu)
cfix=5;      
train_set = cfix*train_set;                                
train_set1 = (cfix-1)*train_set + ones(size(train_set));   
[DL,TL,dsMat,tsMat] =  construct_neighborhood(dis_sim, met_sim);   
[U,V] = AGD_optimization(train_set,train_set1,seed,DL,TL,canshu);    
scores = predict_scores(train_set,dsMat,tsMat,U,V,test_data);
end

function [DL,TL,dsMat,tsMat] = construct_neighborhood(dis_sim, met_sim)
K1 = 5;           
dsMat = dis_sim - diag(diag(dis_sim));      
tsMat = met_sim - diag(diag(met_sim));
if K1 > 0    
    S1 = get_nearest_neighbors(dsMat, K1);   
    DL = laplacian_matrix(S1);
    S2 = get_nearest_neighbors(tsMat, K1);    
    TL = laplacian_matrix(S2);
else  
    DL = laplacian_matrix(dsMat);
    TL = laplacian_matrix(tsMat);
end
end

function X = get_nearest_neighbors(S,K1)
[m, n] = size(S);
X = zeros(m, n);
for i = 1:m
    [~,b] = sort(S(i,:),'descend');
    ii = b(1:min(K1,n));     
    X(i,ii) = S(i, ii);
end
end


function L = laplacian_matrix(S)
x = sum(S);
y = sum(S');
L = 0.5*(diag(x+y) - (S+S')); 
end




function [U,V] = AGD_optimization(train_set,train_set1,seed,DL,TL,canshu)
theta = 1;
num_factors = canshu(1);
[nd,nm] = size(train_set);
max_iter=100;   
if length(seed)==0
    U = sqrt(1/num_factors)*randn(nd,num_factors);
    V = sqrt(1/num_factors)*randn(nm,num_factors);
else
    randn('state',seed)
    U = sqrt(1/num_factors)*randn(nd,num_factors);
    randn('state',seed)
    V = sqrt(1/num_factors)*randn(nm,num_factors);
end
dg_sum = zeros(size(U));
tg_sum = zeros(size(V));
last_log = log_likelihood(U,V,DL,TL,canshu,train_set,train_set1);   
for t =  1:max_iter
    dg =  deriv_opt(train_set,train_set1,U,V,'disease',DL,TL,canshu); 
    dg_sum = dg_sum + dg.^2;  
    vec_step_size = theta*ones(size(dg_sum))./ sqrt(dg_sum);   
    U = U + vec_step_size .* dg;  
    tg = deriv_opt(train_set,train_set1,U,V,'met',DL,TL,canshu);  
    tg_sum = tg_sum + tg.^2;
    vec_step_size = theta*ones(size(tg_sum)) ./ sqrt(tg_sum);
    V = V + vec_step_size .* tg;
    %%%%
    curr_log = log_likelihood(U,V,DL,TL,canshu,train_set,train_set1);   
    delta_log = (curr_log-last_log)/abs(last_log);  
    if abs(delta_log) < 1e-5
        break;
    end
    last_log = curr_log; 
end
end

function vec_deriv = deriv_opt(train_set,train_set1,U,V,name,DL,TL,canshu)
lambda_d = canshu(2);   lambda_t = canshu(2);
alpha = canshu(3);      beta = canshu(3);
if strcmp(name,'disease')==1
    vec_deriv = train_set*V;  
else
    vec_deriv = train_set'*U; 
end
A = U*V';
A = exp(A);
A = A./(A + ones(size(train_set)));   
A = train_set1.* A;     
if strcmp(name,'disease') == 1
    vec_deriv = vec_deriv - A * V;
    vec_deriv = vec_deriv - (lambda_d*U+alpha*DL*U);  
else
    vec_deriv = vec_deriv - A'*U;
    vec_deriv =  vec_deriv - (lambda_t*V+beta*TL*V);
end
end

function loglik = log_likelihood(U,V,DL,TL,canshu,train_set,train_set1)

lambda_d = canshu(2);   lambda_t = canshu(2);
alpha = canshu(3);      beta = canshu(3);

nd = size(U,1);
nm = size(V,1);

loglik = 0;
A = U*V';
B = A .* train_set;  
loglik = loglik + sum(B(:));
A = exp(A);
A =A+ ones(nd,nm);
A = log(A);
A = train_set1 .* A;    
loglik = loglik - sum(A(:));
loglik = loglik - (0.5 * lambda_d * sum(sum(square(U)))+0.5 * lambda_t * sum(sum(square(V))));
loglik = loglik - (0.5 * alpha * sum(sum(diag((U' * DL*U)))));
loglik = loglik - (0.5 * beta * sum(sum(diag((V' * TL*V)))));
end






function scores = predict_scores(train_set,dsMat,tsMat,U,V,test_data)
K2 = 5;          
[x, y] = find(train_set > 0);                              
train_diseases = unique(x);      
train_mets = unique(y);           
K2 = min([K2,length(train_diseases),length(train_mets)]);
dinx = train_diseases;
DS = dsMat(:, dinx);   
tinx = train_mets;
TS = tsMat(:, tinx);   
Nt = size(test_data,1);    

scores = zeros(Nt,1);
for tt = 1:Nt
    d = test_data(tt,1);  
    t = test_data(tt,2);
    if sum(ismember(d,train_diseases))==1        
        if sum(ismember(t,train_mets))==1  
            val = sum(U(d,:).*V(t,:));  
        else    
            [~,index] = sort(TS(t,:),'descend');   
            jj = index(1:K2);     
            val = sum(U(d,:).*(TS(t,jj)*V(tinx(jj), :)))/sum(TS(t, jj));   
        end
    else  
        if sum(ismember(t,train_mets))==1   
            [~,index] = sort(DS(d,:),'descend');   
            ii = index(1:K2);  
            val = sum((DS(d,ii)*U(dinx(ii), :)).*V(t,:))/sum(DS(d, ii));
        else 
            [~,index] = sort(DS(d,:),'descend');
            ii = index(1:K2);   
            
            [~,index] = sort(TS(t,:),'descend');  
            jj = index(1:K2);  
            
            v1 =  DS(d,ii)*U( dinx(ii), :)./sum(DS(d, ii));   
            v2 = TS(t,jj)*V(tinx(jj), :)./sum(TS(t, jj));     
            val = sum(v1.*v2);
        end
    end   
    scores(tt) = exp(val)/(1+exp(val));
end
end




