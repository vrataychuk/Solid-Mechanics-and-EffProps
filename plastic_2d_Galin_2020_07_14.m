clear,figure(1),clf,colormap jet
% Physics
K0       = 1;       %  bulk modulus, Pa
G0       = 1;       % shear modulus, Pa
coh0     = G0*0.01; % Yield stress, Pa
rad0     = 1;       % Radius of the hole
Lx       = 20*rad0; % model length in x
Ly       = 1*Lx;    % model length in y
% Boundary conditions
P_in     = 0.0*coh0;
P_inf    = 2.0*coh0;
tau_inf  =-0.2*coh0;
%loading
dStr      = -0.0005*[ 0.0  1 0]; % from Galin to shear bands
% Numerical grid
ngrid     = 2;
nx       = 32*ngrid;        % number of nodes in x-direction
ny       = nx;         % number of nodes in y-direction
% preprocessing
dx       = Lx/(nx-1);
dy       = Ly/(ny-1);
[x,y]    = ndgrid(-Lx/2:dx:Lx/2  ...
    ,             -Ly/2:dy:Ly/2);
[sxx,syy,sxy,x_B,y_B] =  Galin_exact(coh0,P_inf,P_in,tau_inf,rad0,x,y);
Pr       = -(sxx + syy)/2;
sII      = sqrt((sxx - syy).^2/4 + sxy.^2);
Txx      = sxx + Pr;
Tyy      = syy + Pr;
Tauxy      = av4(sxy);
% numerics
CFL       = 1/2;                     % V Courant Friedrich Levy codition
damp      = 5/1;                     % damping of acoustic waves
nout      = 10;                   % plot every nout
niter     =  1000;                   % max number of iterations
% eiter     = 1e-16;                   % exit criteria from iterations
nt        = 1;
% preprocessing
dt        = 1;                           % time step
rho0      = dt^2/CFL^2/dx^2*(K0+4/3*G0); % inertial density
rad       = sqrt(x.^2+y.^2);
radc      = sqrt(av4(x).^2+av4(y).^2);
% initial conditions
Vx             = zeros(nx+1,ny  );
Vy             = zeros(nx  ,ny+1);
Txye           = zeros(nx  ,ny  );
Str            = 0*dStr;
% action
error          = zeros(niter,1);
for it = 1:nt
    % loading
    Str            = Str     + dStr;
    divV0          = dStr(1) + dStr(2);
    Pr             = Pr  -   K0*divV0;
    Txx            = Txx + 2*G0*(dStr(1)-divV0/3);
    Tyy            = Tyy + 2*G0*(dStr(2)-divV0/3);
    Tauxy            = Tauxy + 2*G0*dStr(3);
    Pr_old         = Pr;
    Txx_old        = Txx;
    Tyy_old        = Tyy;
    Txy_old        = Tauxy;
    Ux             = 0*Vx;
    Uy             = 0*Vy;    
    Pr(rad<rad0)   = 0;
    Txx(rad<rad0)  = 0;
    Tyy(rad<rad0)  = 0;
    Tauxy(radc<rad0) = 0;
    lama           = zeros(nx  ,ny  );
    err_pl = [];
    fid          = fopen( 'P.dat','wb'); fwrite(fid,    Pr(:),'double'); fclose(fid);
    fid          = fopen( 'Vx.dat','wb'); fwrite(fid,    Vx(:),'double'); fclose(fid);
    fid          = fopen( 'Vy.dat','wb'); fwrite(fid,    Vy(:),'double'); fclose(fid);
    fid          = fopen( 'Ux.dat','wb'); fwrite(fid,    Ux(:),'double'); fclose(fid);
    fid          = fopen( 'tauXX.dat','wb'); fwrite(fid,    Txx(:),'double'); fclose(fid);
    fid          = fopen( 'tauYY.dat','wb'); fwrite(fid,    Tyy(:),'double'); fclose(fid);
    fid          = fopen( 'tauXY.dat','wb'); fwrite(fid,    Tauxy(:),'double'); fclose(fid);
    %fid          = fopen( 'Sxx.dat','wb'); fwrite(fid,    sxx(:),'double'); fclose(fid);
    %fid          = fopen( 'Syy.dat','wb'); fwrite(fid,    syy(:),'double'); fclose(fid);
    system(['nvcc -DNGRID=',int2str(ngrid),' -DNITER=',int2str(niter),' -DK=',num2str(K0,16),...
        ' -DG=',num2str(G0,16),' -DCOH=',num2str(coh0,16),' -DRHO=',num2str(rho0,16),...
        ' -DDX=',num2str(dx,16),' -DDY=',num2str(dy,16),' -DDT=',num2str(dt,16), ...
        ' -DDAMP=',num2str(damp,16),' pg2d.cu']);
%     system(['nvcc -DNGRID=',int2str(ngrid),' -DNITER=',int2str(niter),' -DK0=',num2str(K0),' pg2d.cu']);
    tic,system('a.exe');
    GPU_time=toc
    %GBs(irun) = nx*ny*nt*3*2*8/GPU_time/1e9;
    for iter = 1:niter
        Exx  = diff(Vx,1,1)/dx;
        
        Eyy  = diff(Vy,1,2)/dy;
        
        Exy  = 0.5*(diff(Vx(2:end-1,:),1,2)/dy ...
            +       diff(Vy(:,2:end-1),1,1)/dx);
        divV = Exx   + Eyy;
        Exxd = Exx   - divV/3;
        Eyyd = Eyy   - divV/3;
        
%         fil = fopen('Exxc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx, ny);
%         difftest = Exx - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('testEx')
% 
%         fil = fopen('Eyyc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx, ny);
%         difftest = Eyy - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('testEy')
% 
%         fil = fopen('Exyc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx-1, ny-1);
%         difftest = Exy - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('testExy')
        
        Pr  = Pr   -   K0*dt*divV;               
        Tauxy = Tauxy  + 2*G0*dt*Exy;
        Txx = Txx  + 2*G0*dt*Exxd;
        Tyy = Tyy  + 2*G0*dt*Eyyd;
        
%         fil = fopen('Pc.dat', 'rb');
%         Pc = fread(fil, 'double');
%         fclose(fil);
%         Pc = reshape(Pc, nx, ny);
%         diffP = Pr - Pc;
%         max(abs(diffP(:)))
%         plot(231),imagesc(diffP),axis image,colorbar,title('diffP')
%     
%         fil = fopen('tauXXc.dat', 'rb');
%         tauXXc = fread(fil, 'double');
%         fclose(fil);
%         tauXXc = reshape(tauXXc, nx, ny);
%         difftauXX = Txx - tauXXc;
%         max(abs(difftauXX(:)))
%         plot(231),imagesc(difftauXX),axis image,colorbar,title('difftauXX')
% 
%         fil = fopen('tauYYc.dat', 'rb');
%         tauYYc = fread(fil, 'double');
%         fclose(fil);
%         tauYYc = reshape(tauYYc, nx, ny);
%         difftauYY = Tyy - tauYYc;
%         max(abs(difftauYY(:)))
%         plot(231),imagesc(difftauYY),axis image,colorbar,title('difftauYY')
% 
%         fil = fopen('tauXYc.dat', 'rb');
%         tauXYc = fread(fil, 'double');
%         fclose(fil);
%         tauXYc = reshape(tauXYc, nx-1, ny-1);
%         difftauXY = Tauxy - tauXYc;
%         max(abs(difftauXY(:)))
%         plot(231),imagesc(difftauXY),axis image,colorbar,title('difftauXY')
        
        %%%% Plasticity start
        Txye(2:end-1,2:end-1) = av4(Tauxy);
        
        J2                    = sqrt((Txx.^2 + Tyy.^2)/2 + Txye.^2);        
        lam                   = (1-coh0./J2); lam(lam<0) = 0;
        
        
        
        if sum(lam(:)>0)>0
            Txx  = (1-lam).*Txx;
            Tyy  = (1-lam).*Tyy;
            Txye = (1-lam).*Txye;
            Tauxy  = (1-av4(lam)).*Tauxy;
            J2   = sqrt((Txx.^2 + Tyy.^2)/2 + Txye.^2);
            lama = lama + lam;
        end
        
%     fil = fopen('J2c.dat', 'rb');
%     J2c = fread(fil, 'double');
%     fclose(fil);
%     J2c = reshape(J2c, nx, ny);
%     diffJ2 = J2 - J2c;
%     max(abs(diffJ2(:)))
%     plot(231),imagesc(diffJ2),axis image,colorbar,title('diffJ2')
%     fil = fopen('lamc.dat', 'rb');
%     lamc = fread(fil, 'double');
%     fclose(fil);
%     lamc = reshape(lamc, nx, ny);
%     difflam = lam - lamc;
%     max(abs(difflam(:)))
%     plot(231),imagesc(difflam),axis image,colorbar,title('difflam')
        
        
        

        err_pl(iter) = max(J2(:)./coh0-1);
        %%%% Plasticity end
        Sxx  = -Pr   + Txx;
        Syy  = -Pr   + Tyy;
        
%         fil = fopen('testc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx, ny);
%         difftest = Syy - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('test')

%         fil = fopen('dx.dat', 'rb');
%         dxc = fread(fil, 'double');
%         fclose(fil);
%         diffdx = dx - dxc

        Fx   =diff(Sxx(:,2:end-1),1,1)/dx + diff(Tauxy,1,2)/dy;
        Fy   =diff(Syy(2:end-1,:),1,2)/dy + diff(Tauxy,1,1)/dx;
        
%         fil = fopen('tauXYc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx-1, ny-1);
%         difftest = Tauxy - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('test')
%         
%         fil = fopen('Sxxc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx, ny);
%         difftest = Sxx - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('testSx')
%         
%         fil = fopen('Syyc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx, ny);
%         difftest = Syy - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('testSy')
        
%         fil = fopen('testFc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx-1, ny-2);
%         difftest = Fx - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('testFx')
%         fil = fopen('testc.dat', 'rb');
%         testc = fread(fil, 'double');
%         fclose(fil);
%         testc = reshape(testc, nx-2, ny-1);
%         difftest = Fy - testc;
%         max(abs(difftest(:)))
%         plot(231),imagesc(difftest),axis image,colorbar,title('testFy')
        
        Vx(2:end-1,2:end-1) = (Vx(2:end-1,2:end-1)+dt/rho0*Fx)/(1+damp/nx);
        Vy(2:end-1,2:end-1) = (Vy(2:end-1,2:end-1)+dt/rho0*Fy)/(1+damp/nx);
        Ux                  = Ux + Vx*dt;
%         error(iter)         = (max(abs(Vx(:)))/Lx + max(abs(Vy(:)))/Ly)*dt;
        %if error(iter) < eiter,break,end
%         if mod(iter,nout)==0,loglog(error(1:iter)),drawnow, end
    end
    
    
    %Compare
    
    fil = fopen('Vxc.dat', 'rb');
    Vxc = fread(fil, 'double');
    fclose(fil);
    Vxc = reshape(Vxc, nx + 1, ny);
    diffVx = Vx - Vxc;
    max(abs(diffVx(:)))
    plot(231),imagesc(diffVx),axis image,colorbar,title('diffVX')
    
    fil = fopen('Vyc.dat', 'rb');
    Vyc = fread(fil, 'double');
    fclose(fil);
    Vyc = reshape(Vyc, nx , ny+1);
    diffVy = Vy - Vyc;
    max(abs(diffVy(:)))
    plot(231),imagesc(diffVy),axis image,colorbar,title('diffVy')
    
    fil = fopen('Uxc.dat', 'rb');
    Uxc = fread(fil, 'double');
    fclose(fil);
    Uxc = reshape(Uxc, nx + 1, ny);
    diffUx = Ux - Uxc;
    max(abs(diffUx(:)))
    plot(231),imagesc(diffUx),axis image,colorbar,title('diffUx')
    
    fil = fopen('Pc.dat', 'rb');
    Pc = fread(fil, 'double');
    fclose(fil);
    Pc = reshape(Pc, nx, ny);
    diffP = Pr - Pc;
    max(abs(diffP(:)))
    plot(231),imagesc(diffP),axis image,colorbar,title('diffP')
    
    fil = fopen('tauXXc.dat', 'rb');
    tauXXc = fread(fil, 'double');
    fclose(fil);
    tauXXc = reshape(tauXXc, nx, ny);
    difftauXX = Txx - tauXXc;
    max(abs(difftauXX(:)))
    plot(231),imagesc(difftauXX),axis image,colorbar,title('difftauXX')
    
    fil = fopen('tauYYc.dat', 'rb');
    tauYYc = fread(fil, 'double');
    fclose(fil);
    tauYYc = reshape(tauYYc, nx, ny);
    difftauYY = Tyy - tauYYc;
    max(abs(difftauYY(:)))
    plot(231),imagesc(difftauYY),axis image,colorbar,title('difftauYY')
    
    fil = fopen('tauXYc.dat', 'rb');
    tauXYc = fread(fil, 'double');
    fclose(fil);
    tauXYc = reshape(tauXYc, nx-1, ny-1);
    difftauXY = Tauxy - tauXYc;
    max(abs(difftauXY(:)))
    plot(231),imagesc(difftauXY),axis image,colorbar,title('difftauXY')
    
    fil = fopen('tauxyAVc.dat', 'rb');
    tauXYc = fread(fil, 'double');
    fclose(fil);
    tauXYc = reshape(tauXYc, nx, ny);
    difftauXY = Txye - tauXYc;
    max(abs(difftauXY(:)))
    plot(231),imagesc(difftauXY),axis image,colorbar,title('difftauxyAV')
    
    fil = fopen('J2c.dat', 'rb');
    J2c = fread(fil, 'double');
    fclose(fil);
    J2c = reshape(J2c, nx, ny);
    diffJ2 = J2 - J2c;
    max(abs(diffJ2(:)))
    plot(231),imagesc(diffJ2),axis image,colorbar,title('diffJ2')
    
    fil = fopen('lamc.dat', 'rb');
    lamc = fread(fil, 'double');
    fclose(fil);
    lamc = reshape(lamc, nx, ny);
    difflam = lam - lamc;
    max(abs(difflam(:)))
    plot(231),imagesc(difflam),axis image,colorbar,title('difflam')
    
    fil = fopen('lamac.dat', 'rb');
    lamac = fread(fil, 'double');
    fclose(fil);
    lamac = reshape(lamac, nx, ny);
    difflama = lama - lamac;
    max(abs(difflama(:)))
    plot(231),imagesc(difflama),axis image,colorbar,title('difflama')
    
    fil = fopen('Sxxc.dat', 'rb');
    testc = fread(fil, 'double');
    fclose(fil);
    testc = reshape(testc, nx, ny);
    difftest = Sxx - testc;
    max(abs(difftest(:)))
    plot(231),imagesc(difftest),axis image,colorbar,title('testSx')

    fil = fopen('Syyc.dat', 'rb');
    testc = fread(fil, 'double');
    fclose(fil);
    testc = reshape(testc, nx, ny);
    difftest = Syy - testc;
    max(abs(difftest(:)))
    plot(231),imagesc(difftest),axis image,colorbar,title('testSy')

    fil = fopen('Fc.dat', 'rb');
    testc = fread(fil, 'double');
    fclose(fil);
    testc = reshape(testc, nx-1, ny-2);
    difftest = Fx - testc;
    max(abs(difftest(:)))
    plot(231),imagesc(difftest),axis image,colorbar,title('testFx')
    
    fil = fopen('Fyc.dat', 'rb');
    testc = fread(fil, 'double');
    fclose(fil);
    testc = reshape(testc, nx-2, ny-1);
    difftest = Fy - testc;
    max(abs(difftest(:)))
    plot(231),imagesc(difftest),axis image,colorbar,title('testFy')

    fil = fopen('Exxc.dat', 'rb');
    testc = fread(fil, 'double');
    fclose(fil);
    testc = reshape(testc, nx, ny);
    difftest = Exx - testc;
    max(abs(difftest(:)))
    plot(231),imagesc(difftest),axis image,colorbar,title('testEx')

    fil = fopen('Eyyc.dat', 'rb');
    testc = fread(fil, 'double');
    fclose(fil);
    testc = reshape(testc, nx, ny);
    difftest = Eyy - testc;
    max(abs(difftest(:)))
    plot(231),imagesc(difftest),axis image,colorbar,title('testEy')
    
    fil = fopen('Exyc.dat', 'rb');
    testc = fread(fil, 'double');
    fclose(fil);
    testc = reshape(testc, nx-1, ny-1);
    difftest = Exy - testc;
    max(abs(difftest(:)))
    plot(231),imagesc(difftest),axis image,colorbar,title('testExy')
    




    F  = J2/coh0-1; F(F<-coh0*0.01) = NaN;
    lam = lama; lam(lam<0) = 0;
    
    subplot(231),imagesc(J2),axis image,colorbar,title(it)
    subplot(232),imagesc(Pr),axis image,colorbar,title('Pr')
    subplot(233),imagesc(Ux),axis image,colorbar,title('Ux')
    subplot(234),imagesc(log10(lam)),axis image,colorbar,title('lam')
    subplot(235),imagesc(Txx),axis image,colorbar,title('Txx')
    %dP    = abs(Pr((end+1)/2,(end+1)/2) - Pr(1,(end+1)/2))/coh0;
    T_inf = abs(Txx(1,1))/coh0;
    %subplot(236),plot(max(abs(Str)),dP,'*',max(abs(Str)),10*T_inf,'d'),hold on
%     subplot(236),plot(dP,T_inf,'*'),hold on
    title(max(J2(:)./coh0-1)),drawnow
    iter
end
function A_av = av4(A)
A_av = 0.25*(A(1:end-1,1:end-1) + A(1:end-1,2:end) ...
    +        A(2:end  ,1:end-1) + A(2:end  ,2:end));
end
