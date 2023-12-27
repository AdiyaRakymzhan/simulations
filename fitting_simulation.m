%% Creating simulated data 
clear all
close all

t=[0:0.1:300-0.1];
nsim=1;
myrand=rand(nsim,1)*10;

for oo = 1:nsim, 
    
    % build the input u and ideal kernel g
    rn=rand(300,1)*20;
    t0_u=cumsum(rn);
    t0_u=t0_u(find(t0_u<t(end)));
    
    u=gammafun2(t, t0_u, 1, 1, 0.3);
    
    xg_true = [0  1.7  1.0  3.0];
    g=xg_true(4)*(gammafun2(t,xg_true(1),xg_true(2),xg_true(3),1)-0*gammafun2(t,0,4,2,1));
    
    %g=gammafun2(t,0,2,1,1)-0.2*gammafun2(t,0,4,2,1);
    %xg_true = [0 2 1 1 0 4 2 -0.2];
    
    % build y
    y=myconv(u,g/sum(g(:)));
%     y=myconv(u,g/600);
    noise=0.05*randn(size(y));
    
    figure(1), clf,
    subplot(411), plot(t,u+noise),title('Neural activity')
    subplot(412), plot(t,g),title('Gamma function')
    subplot(413), plot(t,y),title('Noiseless vascular activity')
    subplot(414), plot(t,y+noise),title('Noisy vascular activity')
    drawnow,

%% ideal deconvolution
    
    % build the filter
    hf_filt = [ones(300,1); zeros(2400,1); ones(300,1)];
    
    % deconvolve filtered vascular signal
    hh = ifft( hf_filt.*(fft(y+noise)./fft(u)) );
    
%     figure(2), clf,
%     subplot(311), plot(abs(fft(y))), axis tight, grid on,
%     subplot(312), plot(abs(fft(u))), axis tight, grid on,
%     subplot(313), plot(real(hh)), axis tight, xlim([0 300]),
%     drawnow,
    
%     y_noiseless=real(ifft(hf_filt.*(fft(y+noise))));
%     figure; plot(y_noiseless)
    
    hh_all(:,oo)=hh(:);
    
    %% Fitting gamma
    tmpg1=[0 1 1 myrand(oo)];
    tmpgs1=gammafit2_scr(y,[t(:)-t(1) u],tmpg1,[1 2 3 4],[],[],1);
    tmpgs1n=gammafit2_scr(y+noise,[t(:)-t(1) u],tmpg1,[1 2 3 4],[],[],1);
    
    tmpgs1_all(oo)=tmpgs1;
    tmpgs1n_all(oo)=tmpgs1n;
    
    figure(5), clf,
    subplot(311), plot(t,u), title('Neural activity')
    subplot(312), plot(t,tmpgs1n_all(1).yh), xlim([0 20]), title('Gamma function')
    subplot(313), plot(t,y+noise,t,tmpgs1n_all(1).yf,'k'), title('Vascular estimation')
    drawnow,

    xx1_all(oo,:)=tmpgs1.xfinal;
    xx1n_all(oo,:)=tmpgs1n.xfinal;
    
end


figure(4),
subplot(321), plot([tmpgs1n_all(:).r],'x'), ylabel('Model R-val'),
subplot(322), plot(myrand,[tmpgs1n_all(:).r], 'x'), ylabel('Model R-val'), xlabel('Initial Ampl'),
subplot(323), plot([1:nsim],ones(nsim,1)*xg_true(4),'o',[1:nsim],xx1_all(:,4),'x'), ylabel('Model Ampl'), legend('True','Calc'),
subplot(324), plot([1:nsim],ones(nsim,1)*xg_true(2),'o',[1:nsim],xx1_all(:,2),'x'), ylabel('Model Peak'), legend('True','Calc'),
resd=ones(nsim,1)*xg_true(4)-xx1_all(:,4);
subplot(325), plot([1:nsim],resd,'o'),ylabel('Residual'), title('Mean difference:',mean(resd,1))

figure(5),
subplot(211), plot(t(1:600),real(hh_all(1:600,:))), axis tight, grid on,
subplot(212), plot(t(1:600),mean(real(hh_all(1:600,:)),2)), axis tight, grid on,

xg_true
tmpgs1.xfinal
tmpgs1n.xfinal

% mse1=(sum([xx1_all - repmat(xg_true,[1 size(xx1_all,2)])].^2),1);
% 
% mse1n=(sum([xx1n_nall-repmat(xg_true,[1 size(xx1n_all,2)])].^2),1);

%% Minimizers    
% build the input u and ideal kernel g
rn=rand(300,1)*20;
t=[0:0.1:300-0.1];
t0_u=cumsum(rn);
t0_u=t0_u(find(t0_u<t(end)));

u=gammafun2(t, t0_u, 1, 1, 0.3);

xg_true = [0  1.7778  1.0  3.12];
g=xg_true(4)*(gammafun2(t,xg_true(1),xg_true(2),xg_true(3),1)-0*gammafun2(t,0,4,2,1));

%g=gammafun2(t,0,2,1,1)-0.2*gammafun2(t,0,4,2,1);
%xg_true = [0 2 1 1 0 4 2 -0.2];

% build y
y=myconv(u,g/sum(g(:)));
noise=0.05*randn(size(y));

figure(1), clf,
    subplot(411), plot(t,u),title('Neural activity')
    subplot(412), plot(t,g),title('Gamma function')
    subplot(413), plot(t,y),title('Noiseless vascular activity')
    subplot(414), plot(t,y+noise),title('Noisy vascular activity')
    drawnow,

%% 1d minimizer for parameter 4
clear mse1
nsteps=1000;
% xp2_steps=([0:nsteps-1]/nsteps)*2+1;
% xp2_steps=([0:nsteps-1]/nsteps)*4+4/nsteps;
% xp2_steps=linspace(-4,4, nsteps); % amp
% xp2_steps=linspace(0.4,7, nsteps); % b
% xp2_steps=linspace(0.1,3.5, nsteps); % a
xp2_steps=linspace(-1,3, nsteps); % t

t=[0:0.1:300-0.1];

% how we find the optimal kernel
for pp=1:nsteps,
    tmp_xg = [0 0.9883 1.9790 xp2_steps(pp)];
%         tmp_xg = [0.1502 0.9883 1.9790 0.0280];

    tmp_g=tmp_xg(4)*(gammafun2(t,tmp_xg(1),tmp_xg(2),tmp_xg(3),1)-0*gammafun2(t,0,4,2,1));

%     g=gammafun2(t,0,2,1,1)-0.2*gammafun2(t,0,4,2,1);
    %xg_true = [0 2 1 1 0 4 2 -0.2];

    % build y
%     tmp_y=myconv(u,tmp_g/sum(tmp_g(:)));
    tmp_y=myconv(u,tmp_g);
    mse1(pp)=mean([y+noise - tmp_y].^2);
end

figure(5), clf,
plot(xp2_steps,mse1), axis tight, grid on,

xp2_final = xp2_steps(find(mse1==min(mse1)));

% [xg_true(2) xp2_final],

%% 2d minimizer for parameter 2 and 4
clear mse1
nsteps=100;
xp2_steps=linspace(0.1,3.5, nsteps); % a
% xp4_steps=([0:nsteps-1]/nsteps)*2.5+1;
xp4_steps=linspace(0.1,2, nsteps); % amp

for pp2=1:nsteps,
    for pp4=1:nsteps;
        tmp_xg = [0  1  1.9790  xp4_steps(pp4)];
        tmp_g=tmp_xg(4)*(gammafun2(t,tmp_xg(1),tmp_xg(2),tmp_xg(3),1)-0*gammafun2(t,0,4,2,1));

        %g=gammafun2(t,0,2,1,1)-0.2*gammafun2(t,0,4,2,1);
        %xg_true = [0 2 1 1 0 4 2 -0.2];

        % build y
        tmp_y=myconv(u,tmp_g/sum(tmp_g(:)));
%         tmp_y=myconv(u,tmp_g/600);
        mse1(pp2,pp4)=mean( [y+noise - tmp_y].^2 );
    end 
end

%%
figure(6), clf,
plot3(xp2_steps,xp4_steps,mse1(1:50,:)), axis tight, grid on, xlabel('xp2'), ylabel('xp4'), zlabel('mse')

figure(5), clf,
plot(xp2_steps,mse1(:,1:20)), axis tight, grid on,

% Find minimum mean square error and its indices
[minMSE, minIndices] = min(mse1(:));
[minIndex1, minIndex2] = ind2sub([nsteps, nsteps], minIndices);

% Get the optimal values for the second and fourth parameters
xp2_final = xp2_steps(minIndex1);
xp4_final = xp4_steps(minIndex2);
optimalParameter2 = xp2_final;
optimalParameter4 = xp4_final;

% Display the optimal parameter values and the corresponding minimum MSE
disp('Optimal Parameter 2: ' + string(optimalParameter2));
disp('Optimal Parameter 4: ' + string(optimalParameter4));
disp('Minimum MSE: ' + string(minMSE));

%% 4d minimizer for all four parameters
clear mse1
nsteps=20;
xp_steps=([0:nsteps-1]/nsteps);

mse1=zeros(nsteps,nsteps,nsteps,nsteps);

for pp1=1:nsteps,
    for pp2=1:nsteps;
        for pp3=1:nsteps;
            for pp4=1:nsteps;
                tmp_xg = [xp_steps(pp1)*4  xp_steps(pp2)*2+1  xp_steps(pp3)+1/nsteps  xp_steps(pp4)*70+1/nsteps];
                tmp_g=tmp_xg(4)*(gammafun2(t,tmp_xg(1),tmp_xg(2),tmp_xg(3),1)-0*gammafun2(t,0,4,2,1));
           
                % build y
                tmp_y=myconv(u,tmp_g/600);
                mse1(pp1, pp2, pp3, pp4)=mean( [y(:) - tmp_y(:)].^2 );
            end 
        end
    end 
end

% Find minimum mean square error and its indices
[minMSE, minIndices] = min(mse1(:));
[minIndex1, minIndex2, minIndex3, minIndex4] = ind2sub([nsteps, nsteps, nsteps, nsteps,], minIndices);

% Get the optimal values for the second and fourth parameters
xp1_final = xp_steps(minIndex1);
xp2_final = xp_steps(minIndex2);
xp3_final = xp_steps(minIndex3);
xp4_final = xp_steps(minIndex4);
optimalParameter1 = xp1_final;
optimalParameter2 = xp2_final;
optimalParameter3 = xp3_final;
optimalParameter4 = xp4_final;

figure(5), clf,
plot(xp_steps,mse1(:,4)), axis tight, grid on,

% Display the optimal parameter values and the corresponding minimum MSE
disp('Optimal Parameter 1: ' + string(optimalParameter1));
disp('Optimal Parameter 2: ' + string(optimalParameter2));
disp('Optimal Parameter 3: ' + string(optimalParameter3));
disp('Optimal Parameter 4: ' + string(optimalParameter4));
disp('Minimum MSE: ' + string(minMSE));

%% 1d minimizer for parameter 6
clear tmp_xg tmp_g tmp_y mse1 xp2_final

nsteps=10000;
    
% build the input u and ideal kernel g
rn=rand(300,1)*20;
t0_u=cumsum(rn);
t0_u=t0_u(find(t0_u<t(end)));

u=gammafun2(t, t0_u, 1, 1, 0.3);
xg_true = [0.5 1.5 1 -0.8];
g=gammafun2(t,0,1,1,0.3)-xg_true(4)*gammafun2(t,xg_true(1),xg_true(2),xg_true(3),1);

% build y
y=myconv(u,g/sum(g(:)));
noise=0.5*randn(size(y));

xp2_steps=([0:nsteps-1]/nsteps)*2+1;

% how we find the optimal kernel
for pp=1:nsteps,
    tmp_xg = [0.5  xp2_steps(pp)  1.0  -0.8];
    tmp_g=gammafun2(t,0,1,1,0.3)-tmp_xg(4)*gammafun2(t,tmp_xg(1),tmp_xg(2),tmp_xg(3),1);
    
    % build y
    tmp_y=myconv(u,tmp_g/sum(tmp_g(:)));
    mse1(pp)=mean( [y - tmp_y].^2 );
end

figure(5), clf,
plot(xp2_steps,mse1), axis tight, grid on,

xp2_final = xp2_steps(find(mse1==min(mse1)));
xp2_final = xp2_steps(find(mse1==min(mse1)));

[xg_true(2) xp2_final],


%% Fitting gamma
%     tmpg1=[0 optimalParameter2 1 optimalParameter4];
    tmpg1=[0 1 1 0.3];
    tmpgs1=gammafit2_scr(y,[t(:)-t(1) u],tmpg1,[1 2 3 4],[],[],1);
    
    tmpg2=[0 xp2_final 1 0.3];
    tmpgs2=gammafit2_scr(y,[t(:)-t(1) u],tmpg2,[1 2 3 4],[],[],1);

    tmpgs1n=gammafit2_scr(y+noise,[t(:)-t(1) u],tmpg1,[1 2 3 4],[],[],1);
    
    figure(3), clf,
    subplot(311), plot(t,u), title('Neural activity')
    subplot(312), plot(t,y), title('Vascular activity')
%     subplot(312), plot(t,tmpgs1.yh), xlim([0 20]), title('Gamma function')
    subplot(313), plot(t,y),
    hold on 
    plot (t, tmpgs1.yf), title('Vascular estimation'), legend
    drawnow,
    