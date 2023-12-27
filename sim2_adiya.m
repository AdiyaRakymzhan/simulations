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
xg_true = [0  1.2134 1.0  1.7856];
g=gamma_CMR(t,xg_true(1),xg_true(2),xg_true(3),xg_true(4));

% build y
y=myconv(u+noiseu,g/sum(g(:)));
% y=myconv(u,g/150);
noisey=0.1*randn(size(y));

% u=u'; noiseu=noiseu';
y=y'; noisey=noisey';

    figure(1), clf,
    subplot(311), plot(t,u+noiseu),title('Neural activity')
    subplot(312), plot(t,g),title('Gamma function')
    subplot(313), plot(t,y+noisey),title('Vascular activity')
    drawnow,
    
%% 1D minimizer
clear mse1 tmp_xg tmpg1 tmp_g tmpgs1 tmpgs1n tmpgs1_all tmpgs1n_all  xx1_all xx1n_all rr

nsteps=100;

xp_steps=linspace(0.1,1.8, nsteps); 

for oo=1:nsteps,
    tmpg1=[0 1 1 xp_steps(oo)];
    tmpgs1n=gammafit2_scr_adiya(y+noisey,[t(:)-t(1) u+noiseu],tmpg1,[1 2 3 4],[],1,1);
    tmpgs1n_all(oo,:)=tmpgs1n;
    mse1(oo)=mean([y+noisey - tmpgs1n.yf].^2);
end

[minValue, minIndex] = min(mse1(:));

xp_final = xp_steps(minIndex);

fin_parms=[0 1.2134 1 xp_final];
tmpg_fin=gamma_CMR(t,fin_parms(1),fin_parms(2),fin_parms(3),fin_parms(4));
tmpy_fin=myconv(u+noiseu,tmpg_fin/sum(tmpg_fin(:)));

% rr=corr(y',tmpy_fin');

figure(2), clf,
    subplot(311), plot(t,u+noiseu), title('Neural activity')
    subplot(312), plot(t,tmpg_fin), xlim([0 20]), title('Gamma function')
    subplot(313), plot(t,y+noisey,t,tmpy_fin,'k'), title('Vascular estimation from noisy data, rr=')
    drawnow,
    
figure(3), clf,
plot(xp_steps,mse1), axis tight, grid on, 
xlabel('xp'), ylabel('mse1'), zlabel('mse')

xg_true 
fin_parms

%% 2D minimizer
clear mse1 tmp_xg tmpg1 tmp_g tmpgs1 tmpgs1n tmpgs1_all tmpgs1n_all  xx1_all xx1n_all rr

nsteps=100;

xp2_steps=linspace(0.1,3.5, nsteps); % alpha
xp4_steps=linspace(0.1,3.5, nsteps); % amp

for oo=1:nsteps, 
    for ss=1:nsteps,
        tmpg1=[0 xp2_steps(ss) 1 xp4_steps(oo)];
        tmpgs1n=gammafit2_scr_adiya(y+noisey,[t(:)-t(1) u+noiseu],tmpg1,[1 2 3 4],[],[],1);
        mse1(ss,oo)=mean([y - tmpgs1n.yf].^2);
    end
end

[minValue, minIndex] = min(mse1(:));
[ind4, ind2] = ind2sub(size(mse1), minIndex);

xp2_final(1) = xp2_steps(ind2);
xp2_final(2) = find(xp2_steps==xp2_final(1));

xp4_final(1) = xp4_steps(ind4);
xp4_final(2) = find(xp4_steps==xp4_final(1));

fin_parms=[0 xp2_final(1) 1 xp4_final(1)];
tmpg_fin=gamma_CMR(t,fin_parms(1),fin_parms(2),fin_parms(3),fin_parms(4));
tmpy_fin=myconv(u+noiseu,tmpg_fin/sum(tmpg_fin(:)));

rr=corr(y,tmpy_fin');

figure(2), clf,
    subplot(311), plot(t,u+noiseu), title('Neural activity')
    subplot(312), plot(t,tmpg_fin), xlim([0 100]), title('Gamma function')
    subplot(313), plot(t,y+noisey,t,tmpy_fin,'k'), title('Vascular estimation from noisy data, rr=',rr)
    drawnow,
    
figure(3), clf,
mesh(xp2_steps,xp4_steps,mse1), axis tight, grid on, 
xlabel('xp2'), ylabel('xp4'), zlabel('mse')

xg_true
fin_parms

%% 3D minimizer
clear mse1 tmp_xg tmpg1 tmp_g tmpgs1 tmpgs1n tmpgs1_all tmpgs1n_all  xx1_all xx1n_all rr

nsteps=100;

xp2_steps=linspace(0.1,3, nsteps); % alpha
xp3_steps=linspace(0.1,2, nsteps);
xp4_steps=linspace(0.5,4, nsteps); % amp

for oo=1:nsteps, 
    for kk=1:nsteps,
        for ss=1:nsteps,
            tmpg1=[0 xp2_steps(ss) xp3_steps(kk) xp4_steps(oo)];
            tmpgs1n=gammafit2_scr_adiya(y+noisey,[t(:)-t(1) u+noiseu],tmpg1,[1 2 3 4],[],[],1);
            mse1(ss,kk,oo)=mean([y - tmpgs1n.yf].^2);
        end
    end
end

[minValue, minIndex] = min(mse1(:));
[ind4, ind3, ind2] = ind2sub(size(mse1), minIndex);

xp2_final(1) = xp2_steps(ind2);
xp2_final(2) = find(xp2_steps==xp2_final(1));

xp3_final(1) = xp3_steps(ind3);
xp3_final(2) = find(xp3_steps==xp3_final(1));

xp4_final(1) = xp4_steps(ind4);
xp4_final(2) = find(xp4_steps==xp4_final(1));

fin_parms=[0 xp2_final(1) xp3_final(1) xp4_final(1)];
tmpg_fin=gamma_CMR(t,fin_parms(1),fin_parms(2),fin_parms(3),fin_parms(4));
tmpy_fin=myconv(u+noiseu,tmpg_fin/sum(tmpg_fin(:)));

rr=corr(y,tmpy_fin');

figure(2), clf,
    subplot(311), plot(t,u+noiseu), title('Neural activity')
    subplot(312), plot(t,tmpg_fin), xlim([0 100]), title('Gamma function')
    subplot(313), plot(t,y+noisey,t,tmpy_fin,'k'), title('Vascular estimation from noisy data, rr=',rr)
    drawnow,

figure(3), clf,
mesh(xp2_steps,xp3_steps,squeeze(mse1(:,:,1))), axis tight, grid on, 
xlabel('xp2'), ylabel('xp3'), zlabel('mse')

figure(4), clf,
mesh(xp2_steps,xp4_steps,squeeze(mse1(:,1,:))), axis tight, grid on, 
xlabel('xp2'), ylabel('xp4'), zlabel('mse')

figure(5), clf,
mesh(xp3_steps,xp4_steps,squeeze(mse1(1,:,:))), axis tight, grid on, 
xlabel('xp3'), ylabel('xp4'), zlabel('mse')

xg_true
fin_parms