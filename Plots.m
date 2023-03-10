%Numerical evaluation of HS density as Laplace and normal mixtures
%Anindya Bhadra, April 2021
clear
x=[-2:0.03:2];
tau=1;

%Evaluate HS density as a Laplace mixture wrt Dawson function.
fun_exp_mixture=@(u,x, tau) (2./pi).*(1./sqrt(pi)).*(1./tau).*exp(-abs(x).*u./tau).*dawson(u./sqrt(2));
phs=zeros(1,length(x));
for i = 1:length(x)
    phs(i)=integral(@(u) fun_exp_mixture(u,x(i),tau),0,Inf);
end

%Evaluate HS density as a normal mixture wrt half Cauchy.
fun_normal_mixture=@(u,x,tau) (1./sqrt(2.*pi.*u.^2.*tau.^2)).*exp(-x^2./(2.*u.^2.*tau.^2)).*(2./pi).*(1./(1+u.^2));
phs_alt=zeros(1,length(x));
for i = 1:length(x)
    phs_alt(i)=integral(@(u) fun_normal_mixture(u,x(i),tau),0,Inf);
end

%Evaluate HS density 1st derivative as a normal mixture wrt half Cauchy.
fun_normal_mixture_deriv = @(u,x,tau) (1./sqrt(2.*pi.*u.^2.*tau.^2)).*(-x./(u.^2.*tau.^2)).*exp(-x^2./(2.*u.^2.*tau.^2)).*(2./pi).*(1./(1+u.^2));
phs_deriv=zeros(1,length(x));
for i = 1:length(x)
     phs_deriv(i)=integral(@(u) fun_normal_mixture_deriv(u,x(i),tau),0,Inf);
end
     
%Plots. See third panel: Max difference in log scale < 10^{-10}.
figure;
subplot(1,3,1);
plot(x,log(phs),'r','LineWidth',1,'Color',[0 0 0]);
xlabel('x');
ylabel('$\log(p^{}_{HS}(x))$: Laplace mixture','Interpreter','latex');
subplot(1,3,2);
plot(x,log(phs_alt),'g','LineWidth',1,'Color',[0 0 0]);
xlabel('x');
ylabel('$\log(p^{}_{HS}(x))$: Cauchy mixture','Interpreter','latex');
subplot(1,3,3);
plot(x,log(phs_alt) - log (phs),'b','LineWidth',1,'Color',[0 0 0]);
xlabel('x');
ylabel('$\log(p^{}_{HS}(x))$:Cauchy - $\log(p^{}_{HS}(x))$:Laplace','Interpreter','latex');
%subplot(1,4,4);
%plot(x, -(phs_deriv)./phs,'b','LineWidth',1,'Color',[0 0 0]);
%xlabel('x');
%ylabel('$-p^{(1)}_{HS}(x)/p_{HS}(x)$','Interpreter','latex');
