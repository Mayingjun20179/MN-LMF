function main_other(cv_flag) 
%%% cv_flag:  cv_flag=1 represent CV on diseases,
%%%To calculate RWR on disease similarity network
%cv_flag=2 represent CV on metabolites
%%%To calculate PROFANCY (Shang’s method)
%cv_flag=3 represent CV on interaction
%%%To calculate PROFANCY (Shang’s method), WMANRWR (Hu’s method), CFRWR (Wang’s method)

D_M = chuli_opt();  %Import training data
cv = 5; %5-fold cross validation
seeds = [7771,4659,22,8367,1812]; 
if cv_flag==1  %CVd      [2.0000    0.4000   0.6897] 
    option = [2,0.4];
    cv_seed(D_M,seeds,cv_flag,cv,option)
elseif cv_flag==2 %CVm [1.0000    0.1000    0.9248]
    option = [1.0000,0.1000];
    cv_seed(D_M,seeds,cv_flag,cv,option) 
elseif cv_flag==3 %CVa
    %[1.0000    0.9000   0.9247]  PROFANCY
    %[1.0000    0.1000   0.6652]  WMANRWR
    %[1.0000    0.1000   0.6189]  CFRWR
    option = [1.0000    0.9000   0.9247;
        1.0000    0.1000   0.6652;
        1.0000    0.1000   0.6189];
    cv_seed(D_M,seeds,cv_flag,cv,option)   
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

function result = cv_seed(D_M,seeds,cv_flag,cv,option)
Ns = length(seeds); 
% num_factors = option(1);  lambda_d = option(3);
% lambda_t = option(2); alpha = option(3);  beta = option(3);
% K = option(4);  num = option(5);  
result = 0;
for i=1:Ns 
    cv_data = cross_validation(D_M.interaction,seeds(i),cv_flag,cv);     
    result0 = fivecross(D_M,cv_data,option,cv_flag);
    result = result+result0;
end
result = result/Ns;
end

function result = fivecross(D_M,cv_data,option,cv_flag)
cv = length(cv_data); 
if cv_flag <= 2
    result = 0;
else
    result = zeros(3,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:cv
    k
    %%%% cv_flag = 1,To calculate RWR on disease similarity network
    if cv_flag == 1 
        scores = RWR_dis(D_M,cv_data{k},option);
        result0 = model_evaluate(scores,cv_data{k}{3}); 
        result = result + result0; 
    elseif cv_flag == 2
        scores = PROFANCY_opt(D_M,cv_data{k},option);
        result0 = model_evaluate(scores,cv_data{k}{3}); 
        result = result + result0; 
    elseif cv_flag == 3
        %%%%%%%%%%%%%PROFANCY
        scores = PROFANCY_opt(D_M,cv_data{k},option(1,:));
        result0 = model_evaluate(scores,cv_data{k}{3}); 
        result(1,:) = result(1,:) + result0; 
        %%%%%%%%%%%%%WMANRWR
        scores = WMANRWR_opt(D_M,cv_data{k},option(2,:));
        result0 = model_evaluate(scores,cv_data{k}{3}); 
        result(2,:) = result(2,:) + result0;    
        %%%%%%%%%%%%%CFRWR
        scores = CFRWR_opt(D_M,cv_data{k},option(3,:));
        result0 = model_evaluate(scores,cv_data{k}{3}); 
        result(3,:) = result(3,:) + result0; 
    end      
end
result = result/cv;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PROFANCY 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(Shang’s method)
function scores = RWR_dis(D_M,cv_data,option) 
interaction = D_M.interaction;
dis_sim = D_M.dis_sim;
train_set = interaction.*cv_data{1};  
scores_RWR =  RWR_opt(train_set',dis_sim,option(2),option(1))';
scores = arrayfun(@(x,y)tx_opt(scores_RWR,x,y),cv_data{2}(:,1),cv_data{2}(:,2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PROFANCY 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(Shang’s method)
function scores = PROFANCY_opt(D_M,cv_data,option)
interaction = D_M.interaction;
met_sim = D_M.met_sim;
train_set = interaction.*cv_data{1};   
%%%%%%
scores_RWR =  RWR_opt(train_set,met_sim,option(2),option(1));
scores = arrayfun(@(x,y)tx_opt(scores_RWR,x,y),cv_data{2}(:,1),cv_data{2}(:,2));
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PROFANCY 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(Shang’s method)
function scores = WMANRWR_opt(D_M,cv_data,option)
interaction = D_M.interaction;
dis_sim = D_M.dis_sim;
train_set = interaction.*cv_data{1};   
met_sim =  WMAN_sim(train_set,dis_sim);
scores_RWR =  RWR_opt(train_set,met_sim,option(2),option(1));
scores = arrayfun(@(x,y)tx_opt(scores_RWR,x,y),cv_data{2}(:,1),cv_data{2}(:,2));
end

function met_sim =  WMAN_sim(dis_met,dis_sim)
[~,Nm] = size(dis_met); 
met_dis_ind = cell(Nm,1);
for i=1:Nm
    met_dis_ind{i} = find(dis_met(:,i)==1);
end
met_sim = zeros(Nm,Nm);
for i=1:Nm
    sim_col = cellfun(@(x)jisuan_met(x,met_dis_ind{i},dis_sim),met_dis_ind);
    met_sim(i,:) = sim_col;
end
end

function S_m = jisuan_met(indexm1,indexm2,dis_sim)
if length(indexm1)==0 || length(indexm2)==0
    S_m = 0;
else
    S1 = dis_sim(indexm1,indexm2);
    s1 = max(S1,[],2);
    S2 = dis_sim(indexm2,indexm1);
    s2 = max(S2,[],2);
    S_m = (sum(s1)+sum(s2))/(length(indexm1)+length(indexm2));
end
end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CFRWR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(Wang’s method)
function scores = CFRWR_opt(D_M,cv_data,option)
interaction = D_M.interaction;
dis_sim = D_M.dis_sim;
train_set = interaction.*cv_data{1};    
met_sim =  jisuanM_sim(train_set,dis_sim);
scores_RWR =  RWR_opt(train_set,met_sim,option(2),option(1));
scores = arrayfun(@(x,y)tx_opt(scores_RWR,x,y),cv_data{2}(:,1),cv_data{2}(:,2));
end

function met_sim =  jisuanM_sim(dis_met,dis_sim)
[Nd,Nm] = size(dis_met); 
met_dis_ind = cell(Nm,1);
for i=1:Nm
    met_dis_ind{i} = find(dis_met(:,i)==1);
end
met_dis_prop = zeros(Nm,Nd);   
for j=1:Nd   
    sim_col = cellfun(@(x)calculate_met(j,x,dis_sim),met_dis_ind);
    met_dis_prop(:,j) = sim_col;
end
met_sim =  cos_opt(met_dis_prop);
end

function S_m = calculate_met(indexd,indexm,dis_sim)
if ismember(indexd,indexm)==1   
    S_m = 1;
else
    if length(indexm)==0        
        S_m = 0;
    else
        S_m = max(dis_sim(indexd,indexm));
    end
end
end

function W = cos_opt(X)
W = X*X';
DX = sqrt(diag(W))*sqrt(diag(W))';
ind = find(DX>0);
W(ind) = W(ind)./DX(ind);

end
    
function Rt = RWR_opt(A,mir_sim,alpha,r)
%random walk on mirna similarity. 
%A: adjacency matrix 
%normFun: Laplacian normalization for disease similarity and miRNA similarity  
%mir_dism: mirna similarity network

norm_mir = normFun(mir_sim);
%R0: initial probability
R0 = A/sum(A(:));
Rt = R0;
for t=1:r
    Rt = alpha *   Rt * norm_mir  + (1-alpha)*R0;
end

end

function result = normFun( M )
%normFun: Laplacian normalization
num = size(M,1);
nM = zeros(num,num);
result = zeros(num,num);
for i = 1:num
    nM(i,i) = sum(M(i,:));
end
for i = 1:num
    rsum = nM(i,i);
    for j = 1:num
        csum = nM(j,j);
        if((rsum==0)||(csum==0))
            result(i,j) = 0;
        else
            result(i,j) = M(i,j)/sqrt(rsum*csum);
        end
    end
end
    
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


function c = tx_opt(A,b,c)
c = A(b,c);
end