function yy=fitoptim_ar(y,u,t,x0,fit_type,xlb,xub)

y=double(y);
u=double(u);
t=double(t);

% curve with initial parms
y0=gamma_parms(x0,t);

% optimizers
if fit_type==1,
    xopt=optimset('lsqnonlin');
    xopt.TolxFun=1e-10;
    xopt.TolX=1e-8;

    if ~exist('xlb','var'), 
      xlb=[0 0.1 0.95 1]; end
    if ~exist('xub','var'), 
      xub=[0 3 1 2.56]; end
    xx1=lsqnonlin(@gammafit_ar,x0,xlb,xub,xopt,t,y,u);
else
    xopt=optimset('fminsearch');
    xopt.TolxFun=1e-10;
    xopt.TolX=1e-8;
    xx1=fminsearch(@gammafit_ar,x0,xopt,t,y,u);
end 

% fitted curve
y1=gamma_parms(xx1,t);

% output
yy.t=t
yy.u=u;
yy.y=y;
yy.x1=xx1;
yy.gfit=y1;
yy.yfit=myconv(u,y1);
yy.ee=mean((y-yy.yfit).^2);

%%%
function y=gamma_parms(xx,tt)
  y=xx(4)*gammafun2(tt,xx(1),xx(2),xx(3));
return,

