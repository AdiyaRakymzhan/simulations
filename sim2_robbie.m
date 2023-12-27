%% Start 
clear all
close all

% nsim=20;
t=[0:0.1:300-0.1];
% myrand = (rand(1, nsim))*4;

%% Create fake data
% build the input u 
rn=rand(300,1)*20;
t0_u=cumsum(rn);
t0_u=t0_u(find(t0_u<t(end)));
u=gammafun2(t, t0_u, 1, 1, 0.3);
noiseu=0.1*randn(size(u));

% build the ideal kernel g
xg_true = [0  1.7778  1.0  3.12];
g=gamma_CMR(t,xg_true(1),xg_true(2),xg_true(3),xg_true(4));

% build y
y=myconv(u+noiseu,g/sum(g(:)));
% y=myconv(u,g/150);
noisey=0.1*randn(size(y));

    figure(1), clf,
    subplot(311), plot(t,u+noiseu),title('Neural activity')
    subplot(312), plot(t,g),title('Gamma function')
    subplot(313), plot(t,y+noisey),title('Vascular activity')
    drawnow,
    
%% 1D minimizer
clear mse1 tmp_xg tmpg1 tmp_g tmpgs1 tmpgs1n tmpgs1_all tmpgs1n_all  xx1_all xx1n_all rr

nsteps=100;

xp_steps=linspace(0.1,4, nsteps); % alpha

for oo=1:nsteps,
    tmpg1=[0 xp_steps(oo) 1 0.3];
    tmp_g=gamma_CMR(t,tmpg1(1),tmpg1(2),tmpg1(3),tmpg1(4));
    yy=gamma_fit_CMR(y+noisey, u+noiseu,t,tmpg1);
    
%     tmpgs1n_all(oo,:)=tmpgs1n_all.xx;
    xx1n_all(oo,:)=yy.xx;

    mse1(oo)=mean([y+noisey - yy.yfit].^2);
end


[minValue, minIndex] = min(mse1(:));

xp_final = xp_steps(minIndex);

fin_parms=[0 1 1 xp_final];
tmpg_fin=gamma_CMR(t,fin_parms(1),fin_parms(2),fin_parms(3),fin_parms(4));
tmpy_fin=myconv(u+noiseu,tmp_g/sum(tmp_g(:)));

rr=corr(y,tmpy_fin);

figure(2), clf,
    subplot(311), plot(t,u+noiseu), title('Neural activity')
    subplot(312), plot(t,tmpg_fin), xlim([0 20]), title('Gamma function')
    subplot(313), plot(t,y+noisey,t,tmpy_fin,'k'), title('Vascular estimation from noisy data, rr=',rr)
    drawnow,
    
figure(3), clf,
plot(xp_steps,mse1), axis tight, grid on, 
xlabel('xp2'), ylabel('mse1'), zlabel('mse')

xg_true 
fin_parms

%% 2D minimizer
clear mse1 tmp_xg tmpg1 tmp_g tmpgs1 tmpgs1n tmpgs1_all tmpgs1n_all  xx1_all xx1n_all rr

nsteps=10;

xp2_steps=linspace(0.1,10, nsteps); % alpha
xp4_steps=linspace(0.1,10, nsteps); % amp

for oo=1:nsteps, 
    for ss=1:nsteps,
        tmpg1=[0.8081 xp2_steps(ss) 1.7103 xp4_steps(oo)];
        tmp_g=gamma_CMR(t,tmpg1(1),tmpg1(2),tmpg1(3),tmpg1(4));
        tmpgs1n=gammafit2_scr(y+noisey,[t(:)-t(1) u+noiseu],tmpg1,[1 2 3 4],[],[],1);
        mse1(ss,oo)=mean([y - tmpgs1n.yf].^2);
    end
end

[minValue, minIndex] = min(mse1(:));
[ind2, ind4] = ind2sub(size(mse1), minIndex);

xp2_final(1) = xp2_steps(ind2);
xp2_final(2) = find(xp2_steps==xp2_final(1));

xp4_final(1) = xp4_steps(ind4);
xp4_final(2) = find(xp4_steps==xp4_final(1));

fin_parms=[0 xp2_final(1) 1 xp4_final(1)];
tmpg_fin=gamma_CMR(t,fin_parms(1),fin_parms(2),fin_parms(3),fin_parms(4));
tmpy_fin=myconv(u+noiseu,tmp_g/sum(tmp_g(:)));

% rr=corr(y,tmpy_fin);

figure(2), clf,
    subplot(311), plot(t,u+noiseu), title('Neural activity')
    subplot(312), plot(t,tmpg_fin), xlim([0 100]), title('Gamma function')
    subplot(313), plot(t,y+noisey,t,tmpy_fin,'k'), title('Vascular estimation from noisy data, rr=')
    drawnow,
    
figure(3), clf,
mesh(xp2_steps,xp4_steps,mse1), axis tight, grid on, 
xlabel('xp2'), ylabel('xp4'), zlabel('mse')

xg_true
fin_parms

%% Double-gamma 

