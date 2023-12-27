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
g=xg_true(4)*(gammafun2(t,xg_true(1),xg_true(2),xg_true(3),1));

% build y
y=myconv(u+noiseu,g/sum(g(:)));
% y=myconv(u,g/150);
noisey=0.05*randn(size(y));

    figure(1), clf,
    subplot(311), plot(t,u+noiseu),title('Neural activity')
    subplot(312), plot(t,g),title('Gamma function')
    subplot(313), plot(t,y+noisey),title('Vascular activity')
    drawnow,
    
%% Optimization function: fminsearch
clear optim_parms mse1

tmpg_init = [0 1 1 0.3];

% Define the objective function
objective_function = @(x) calculate_mse(t, x, u, noiseu, y, noisey);

% To determine function parameters, perform optimization using fminsearch
options = optimset('Display', 'iter'); % Optional: Display optimization progress
[tmpg_optim, mse_optim] = fminsearch(objective_function, tmpg_init, options);

% Store the optimized parameter values and MSE
optim_parms = tmpg_optim;
mse1 = mse_optim;
   
% Find the index with the minimum MSE
[min_mse, min_mse_index] = min(mse1);

% Retrieve the optimized parameter values corresponding to the minimum MSE
final_parms = optim_parms;

% Display the results
fprintf('Optimal parameter values: [%.2f, %.2f, %.2f, %.2f]\n', final_parms);
fprintf('Minimum MSE: %.4f\n', min_mse);

% Calculate the final predicted vascular signal
tmpg_fin = final_parms(4) * (gammafun2(t, final_parms(1), final_parms(2), final_parms(3), 1));
tmpy_fin = myconv(u+noiseu, tmpg_fin/sum(tmpg_fin(:)));

rr=corr(y+noisey,tmpy_fin);

figure, clf,
    subplot(311), plot(t,u+noiseu), title('Neural activity')
    subplot(312), plot(t,tmpg_fin), xlim([0 100]), title('Gamma function')
    subplot(313), plot(t,y+noisey,t,tmpy_fin,'k'), title('Vascular estimation, rr=', rr), legend('measured', 'estimated')
    drawnow,

% Function to calculate the mean squared error
function mse = calculate_mse(t, x, u, noiseu, y, noisey, lb, ub)
    
% Define the function
    tmp_g = x(4) * (gammafun2(t, x(1), x(2), x(3), 1));
    tmp_y = myconv(u+noiseu, tmp_g/sum(tmp_g(:)));
% Minimize the mean squre difference between the predicted and original
% signal
    mse = mean((y+noisey - tmp_y).^2);
end

