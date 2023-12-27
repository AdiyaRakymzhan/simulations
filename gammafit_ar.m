function mse1=gammafit_ar(x,t,y,u,tbias)

if nargin<5, tbias=[1 length(t)]; end;

t_del=x(1);
a=x(2);
b=x(3);
amp=x(4);

g=amp*gammafun2(t,t_del,a,b);
yf=myconv(u,g/sum(g(:)));

% Penalty 
lb = [0,0.1,0.95,1];  % Lower bounds for the parameters
ub = [0,3,1,2.56];  % Upper bounds for the parameters

if any(x < lb) || any(x > ub)
    % Penalize or discard solutions that violate the bounds
   mse1 = 1e+6;  
    return;
end

mse1=mean((y - yf).^2);

    
disp(sprintf('  a=%.2f, b=%.2f, amp=%.2f, t0=%.2f',x(2),x(3),x(4),x(1)));

if nargout==0, plot(t,yf,t,y), end;

end 