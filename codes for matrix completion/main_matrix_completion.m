%%%%%%%%%%% Test the proposed algorithm: AM_our on MovieLens-1M data %%%%%%
close all; clear;clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = load('movielen1m.mat');
data = file.data;    
data = data/max(abs(data(:)));
[trainY, testY] = splitData(data, 0.7);  %%$70\%$ for training,  $30\%$ for testing.      
[m,n] = size(trainY); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
markTrainY = (trainY ~= 0);
[rowTrainY, colTrainY] = find(markTrainY);
TrainLabs0 = trainY(markTrainY);
TrainLabs = TrainLabs0;
EE   = speye(m+n,m+n);       
A0 = [sparse(m,m),trainY; trainY',sparse(n,n)];
A0 = triu(A0,1);
idxs_A0 = find(A0); 
%%%%%%%%%%%%%%%%%%%%%% Data preprocessing and initialization%%%%%%%%%%%%%%%
opts.maxit = 1*1e2;  %% Maximum number of iterations
opts.tol = 1e-4;     %% Tolerance
opts.eta = 1e-3;     %% Regularization coefficient
opts.gamma0 = 1e0;  %% Initial value of penalty parameter
opts.back   = 3;     %% The stopping criterion is that 
%% the objective value change over three consecutive iterations 
%% falls below 1e-4. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.EE        = EE;
opts.sEE       = speye(length(TrainLabs));
opts.X0        = normrnd(0,1,m+n,2); 
opts.Y0        = normrnd(0,1,m+n,2); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.A0  = A0;
opts.idxs_A0 = idxs_A0;    
opts.Q_l = EE(rowTrainY,:);
opts.Q_r = EE(m+colTrainY,:);  
opts.TrainLabs = TrainLabs;
opts.trainY   = trainY;
opts.testY    = testY;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_our %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[hist_tt, hist_gamma, hist_penalty, hist_testError, hist_obj] = AM_our(opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% drawing results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(1); set(fid,'position',[546   347   304   234]);
loglog(hist_tt, hist_gamma, '-r')
ylabel('penalty parameter $${\gamma}_k$$','FontSize',12,'Interpreter','latex'); 
xlabel('time (s)','FontSize',12);
h = legend('\emph{AM$$_{our}$$}','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(2); set(fid,'position',[546   347   304   234]);
loglog(hist_tt, hist_penalty, '-r');
xlabel('time (s)','FontSize',12);
ylabel('$$\|X_k-Y_k\|$$','FontSize',12,'Interpreter','latex');
h = legend('\emph{AM$$_{our}$$}','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(3); set(fid,'position',[546   347   304   234]);
loglog(hist_tt, hist_testError, '-r');
xlabel('time (s)','FontSize',12);
ylabel('Testing RMSE','FontSize',12,'Interpreter','latex');
h = legend('\emph{AM$$_{our}$$}','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(4); set(fid,'position',[546   347   304   234]);
loglog(hist_tt, hist_obj, '-r')
ylabel('objective value','FontSize',12);
xlabel('time (s)','FontSize',12);
h = legend('\emph{AM$$_{our}$$}','Interpreter','latex');


