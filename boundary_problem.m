clear, figure(1), clf, colormap jet

% PHYSICS
Lx  = 10.0;    % physical length
Ly  = 10.0;    % physical width
K   = 1.0;     % bulk modulus
rho = 1.0;     % density
G   = 0.5;     % shear modulus

% NUMERICS
Nx  = 32;     % number of space steps
Ny  = 32;
Nt  = 10;     % number of time steps
CFL = 0.5;     % Courant�Friedrichs�Lewy

% PREPROCESSING
dX     = Lx / (Nx - 1);                                   % space step
dY     = Ly/(Ny - 1);
x      = (-Lx / 2) : dX : (Lx / 2);                       % space discretization
y      = (-Ly/2) : dY : (Ly/2);
[x, y] = ndgrid(x, y);                                    % 2D mesh
dt     = CFL * min(dX, dY) / sqrt( (K + 4*G/3) / rho);    % time step

% INITIAL CONDITIONS
P     = zeros(Nx, Ny);
P     = exp(-x .* x - y .* y);    % hydrostatic stress (ball part of tensor)
P0    = P;                        % initial hydrostatic stress
Vx    = zeros(Nx + 1, Ny);        % velocity
Vy    = zeros(Nx, Ny + 1);
tauxx = zeros(Nx, Ny);            % deviatoric stress
tauyy = zeros(Nx, Ny);
tauxy = zeros(Nx - 1, Ny - 1);

% INPUT FILES
pa = [dX, dY, dt, K, G, rho];

% parameters
fil = fopen('pa.dat', 'wb');
fwrite(fil, pa(:), 'double');
fclose(fil);

% initial pressure
fil = fopen('P.dat', 'wb');
fwrite(fil, P(:), 'double');
fclose(fil);

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
  % velocity divergence
  divV                = diff(Vx,1,1)/dX + diff(Vy,1,2)/dY;
  P                   = P     + (-divV * K) * dt;
  tauxx               = tauxx + ((diff(Vx,1,1)/dX - divV/3.0) * G * 2.0) * dt;
  tauyy               = tauyy + ((diff(Vy,1,2)/dY - divV/3.0) * G * 2.0) * dt;
  tauxy               = tauxy + ((diff(Vx(2:end-1,:), 1, 2)/dY + diff(Vy(:,2:end-1), 1, 1)/dX) * G) * dt;
  Vx(2:end-1,2:end-1) = Vx(2:end-1,2:end-1) + (diff(-P(:,2:end-1) + tauxx(:,2:end-1), 1, 1)/dX / rho + diff(tauxy,1,2)/dY)    * dt;
  Vy(2:end-1,2:end-1) = Vy(2:end-1,2:end-1) + (diff(-P(2:end-1,:) + tauyy(2:end-1,:), 1, 2)/dY / rho + diff(tauxy,1,1)/dX)    * dt;
% POSTPROCESSING
  %if mod(it, 10) == 0
  %  pcolor(x, y, P)
  %  title(it)
  %  shading flat
  %  colorbar
  %  axis image        % square image
  %  drawnow
  %end
endfor

% GPU CALCULATION
%system(['nvcc -DNGRID=',int2str(ngrid),' -DNT=',int2str(nt),' -DNPARS=',int2str(length(pa)),' Wave2d_2020_06_04.cu']);
%system(['nvcc boundary_problem.cu']);
%system(['a.exe']);

fil = fopen('Pc.dat', 'rb');
Pc = fread(fil, 'double');
fclose(fil);
Pc = reshape(Pc, Nx, Ny);

fil = fopen('Vxc.dat', 'rb');
Vxc = fread(fil, 'double');
fclose(fil);
Vxc = reshape(Vxc, Nx + 1, Ny);

diffP = P - Pc;
diffVx = Vx - Vxc;

% POSTPROCESSING
subplot(2,2,1)
imagesc(P)
colorbar
title('P')
axis image

subplot(2,2,2)
imagesc(diffP)
colorbar
title('Pc')
axis image

subplot(2,2,3)
imagesc(Vx)
colorbar
title('Vx')
axis image

subplot(2,2,4)
imagesc(diffVx)
colorbar
title('Vxc')
axis image