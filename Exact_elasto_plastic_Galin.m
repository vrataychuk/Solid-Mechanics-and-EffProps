clear,figure(1),clf,colormap jet(256)
% Physics
Ys       = 1;     % Yield stress
R0       = 1;     % Radius of the hole
Lx       = 20*R0; % model length in x
Ly       = 1*Lx;  % model length in y
% Boundary conditions
P_in     = 1.0*Ys;
P_inf    =-2.0*Ys;
tau_inf  = 0.2*Ys;
% Numerical grid
nx       = 251;        % number of nodes in x-direction
ny       = 251;        % number of nodes in y-direction
[x,y]    = ndgrid(linspace(-Lx/2,Lx/2,nx) ...
    ,             linspace(-Ly/2,Ly/2,ny));
[sxx,syy,sxy,x_B,y_B] =  Galin_exact(Ys,P_inf,P_in,tau_inf,R0,x,y);
Pr       = -(sxx + syy)/2;
sII      = sqrt((sxx - syy).^2/4 + sxy.^2);
figure(1),clf,colormap jet
subplot(221),pcolor(x,y,sxx+Pr),shading flat,axis image,colorbar
hold on, plot(x_B,y_B,'w'),hold off,axis off,title('Txx')
subplot(222),pcolor(x,y,sxy),shading flat,axis image,colorbar
hold on, plot(x_B,y_B,'w'),hold off,axis off,title('Sxy')
subplot(223),pcolor(x,y,Pr),shading flat,axis image,colorbar
hold on, plot(x_B,y_B,'w'),hold off,axis off,title('Pr')
subplot(224),pcolor(x,y,sII),shading flat,axis image,colorbar
hold on, plot(x_B,y_B,'w'),hold off,axis off,title('sII')
%plot(x(:,(end+1)/2),sxx(:,(end+1)/2),y((end+1)/2,:),syy((end+1)/2,:))

