clear
figure(1)
clf
colormap jet

% PHYSICS
Lx  = 10.0;    % physical length
Ly  = 10.0;    % physical width
K   = 1.0;     % bulk modulus
rho = 1.0;     % density
G   = 0.5;     % shear modulus

% NUMERICS
nGrid = 2;
Nx  = 32 * nGrid;     % number of space steps
Ny  = 32 * nGrid;
Nt  = 10000;     % number of time steps
CFL = 0.5;     % Courant-Friedrichs-Lewy

% PREPROCESSING
dX     = Lx / (Nx - 1);                                   % space step
dY     = Ly / (Ny - 1);
x      = (-Lx / 2) : dX : (Lx / 2);                       % space discretization
y      = (-Ly / 2) : dY : (Ly / 2);
[x, y] = ndgrid(x, y);                                    % 2D mesh
[xUx, yUx] = ndgrid((-(Lx + dX)/2) : dX : ((Lx + dX)/2), (-Ly/2) : dY : (Ly/2));
[xUy, yUy] = ndgrid((-Lx/2) : dX : (Lx/2), (-(Ly+dY)/2) : dY : ((Ly+dY)/2));
dt     = CFL * min(dX, dY) / sqrt( (K + 4*G/3) / rho);    % time step
damp   = 4 / dt / Nx;

% INITIAL CONDITIONS
P0    = zeros(Nx, Ny);            % initial hydrostatic stress
%P0    = exp(-x .* x - y .* y);    % hydrostatic stress (ball part of tensor)
Ux    = zeros(Nx + 1, Ny);        % displacement
Uy    = zeros(Nx, Ny + 1);
Vx    = zeros(Nx + 1, Ny);        % velocity
Vy    = zeros(Nx, Ny + 1);
tauxx = zeros(Nx, Ny);            % deviatoric stress
tauyy = zeros(Nx, Ny);
tauxy = zeros(Nx - 1, Ny - 1);

% BOUNDARY CONDITIONS
dUxdx = 0.0;
dUydy = 0.002;
dUxdy = 0.0;
Ux = Ux + (dUxdx * xUx + dUxdy * yUx);
Uy = Uy + dUydy * yUy;

% INPUT FILES
pa = [dX, dY, dt, K, G, rho, damp];

% parameters
fil = fopen('pa.dat', 'wb');
fwrite(fil, pa(:), 'double');
fclose(fil);

% initial pressure
%fil = fopen('P0.dat', 'wb');
%fwrite(fil, P0(:), 'double');
%fclose(fil);

% initial Vx
%fil = fopen('Vx.dat', 'wb');
%fwrite(fil, Vx(:), 'double');
%fclose(fil);

% initial Vy
%fil = fopen('Vy.dat', 'wb');
%fwrite(fil, Vy(:), 'double');
%fclose(fil);

% CPU CALCULATION
for it = 1 : Nt
  % displacement divergence
  divU = diff(Ux,1,1) / dX + diff(Uy,1,2) / dY;
  
  % constitutive equation - Hooke's law
  P     = P0 - K * divU;
  tauxx = 2.0 * G * (diff(Ux,1,1)/dX - divU/3.0);
  tauyy = 2.0 * G * (diff(Uy,1,2)/dY - divU/3.0);
  tauxy = G * (diff(Ux(2:end-1,:), 1, 2)/dY + diff(Uy(:,2:end-1), 1, 1)/dX);
  
  % motion equation
  dVxdt = diff(-P(:,2:end-1) + tauxx(:,2:end-1), 1, 1)/dX / rho + diff(tauxy,1,2)/dY;
  Vx(2:end-1,2:end-1) = Vx(2:end-1,2:end-1) * (1 - dt * damp) + dVxdt * dt;
  dVydt = diff(-P(2:end-1,:) + tauyy(2:end-1,:), 1, 2)/dY / rho + diff(tauxy,1,1)/dX;
  Vy(2:end-1,2:end-1) = Vy(2:end-1,2:end-1) * (1 - dt * damp) + dVydt * dt;
  
  % displacements
  Ux = Ux + Vx * dt;
  Uy = Uy + Vy * dt;
  
% POSTPROCESSING
%  if mod(it, 100) == 0
%    subplot(2, 1, 1)
%    pcolor(x, y, diff(Ux,1,1)/dX)
%    title(it)
%    shading flat
%    colorbar
%    axis image        % square image
%    
%    subplot(2, 1, 2)
%    pcolor(x, y, diff(Uy,1,2)/dY)
%    title(it)
%    shading flat
%    colorbar
%    axis image        % square image
%    
%    drawnow
%  endif
endfor

% GPU CALCULATION
%system(['nvcc -DNGRID=',int2str(ngrid),' -DNT=',int2str(nt),' -DNPARS=',int2str(length(pa)),' Wave2d_2020_06_04.cu']);
%system(['nvcc boundary_problem.cu']);
%system(['a.exe']);

fil = fopen('Pc.dat', 'rb');
Pc = fread(fil, 'double');
fclose(fil);
Pc = reshape(Pc, Nx, Ny);

fil = fopen('Uyc.dat', 'rb');
Uyc = fread(fil, 'double');
fclose(fil);
Uyc = reshape(Uyc, Nx, Ny + 1);

fil = fopen('tauXYc.dat', 'rb');
tauXYc = fread(fil, 'double');
fclose(fil);
tauXYc = reshape(tauXYc, Nx - 1, Ny - 1);

diffP = P - Pc;
diffUy = Uy - Uyc;
diffTauXY = tauxy - tauXYc;

% POSTPROCESSING
subplot(2,2,1)
imagesc(Uy)
colorbar
title('Uy')
axis image

subplot(2,2,2)
imagesc(diffUy)
colorbar
title('diffUy')
axis image

subplot(2,2,3)
imagesc(P)
colorbar
title('P')
axis image

subplot(2,2,4)
imagesc(diffP)
colorbar
title('diffP')
axis image

GetSigma = [0 0 0];
GetSigma(1) = mean(tauxx(:) - P(:))
GetSigma(2) = mean(tauyy(:) - P(:))
GetSigma(3) = mean(tauxy(:))