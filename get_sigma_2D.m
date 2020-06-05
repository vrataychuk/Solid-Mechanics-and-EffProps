function S = get_sigma_2D(loadValue, loadType)
  figure(1)
  clf
  colormap jet

  % PHYSICS
  Lx  = 16.0;                         % physical length
  Ly  = 16.0;                         % physical width
  E0   = 2.0;                      % Young's modulus
  nu0  = 0.2;                         % Poisson's ratio  
  rho0 = 2.0;                      % density
  K0   = E0 / (3.0 * (1 - 2 * nu0));  % bulk modulus
  G0   = E0 / (2.0 + 2.0 * nu0);      % shear modulus

  % NUMERICS
  Nx  = 200;     % number of space steps
  Ny  = 200;
  Nt  = 4000000;    % number of time steps
  CFL = 0.125;     % Courant–Friedrichs–Lewy

  % PREPROCESSING
  dX     = Lx / (Nx - 1);                                   % space step
  dY     = Ly / (Ny - 1);
  x      = (-Lx / 2) : dX : (Lx / 2);                       % space discretization
  y      = (-Ly / 2) : dY : (Ly / 2);
  [x, y] = ndgrid(x, y);                                    % 2D mesh
  [xUx, yUx] = ndgrid((-(Lx + dX)/2) : dX : ((Lx + dX)/2), (-Ly/2) : dY : (Ly/2));
  [xUy, yUy] = ndgrid((-Lx/2) : dX : (Lx/2), (-(Ly+dY)/2) : dY : ((Ly+dY)/2));
  dt     = CFL * min(dX, dY) / sqrt( (K0 + 4*G0/3) / rho0);    % time step
  damp   = 4.0 / dt / Nx;
  
  % MATERIALS
  E = zeros(Nx, Ny);
  nu = zeros(Nx, Ny);
  [E, nu] = set_mats_2D(Nx, Ny, x, y);     % Young's modulus and Poisson's ratio
  K = E ./ (3.0 * (1 - 2 * nu));             % bulk modulus
  G = E ./ (2.0 + 2.0 * nu);                 % shear modulus

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
  dUxdx = loadValue * loadType(1);
  dUydy = loadValue * loadType(2);
  dUxdy = loadValue * loadType(3);
  Ux = Ux + (dUxdx * xUx + dUxdy * yUx);
  Uy = Uy + dUydy * yUy;

  % ACTION LOOP
  for it = 1 : Nt
    % displacement divergence
    divU = diff(Ux,1,1) / dX + diff(Uy,1,2) / dY;
    
    % constitutive equation - Hooke's law
    P     = P0 - K .* divU;
    tauxx = 2.0 * G .* (diff(Ux,1,1)/dX - divU/3.0);
    tauyy = 2.0 * G .* (diff(Uy,1,2)/dY - divU/3.0);
    tauxy = av4(G) .* (diff(Ux(2:end-1,:), 1, 2)/dY + diff(Uy(:,2:end-1), 1, 1)/dX);
    
    % motion equation
    dVxdt = diff(-P(:,2:end-1) + tauxx(:,2:end-1), 1, 1)/dX / rho0 + diff(tauxy,1,2)/dY;
    Vx(2:end-1,2:end-1) = Vx(2:end-1,2:end-1) * (1 - dt * damp) + dVxdt * dt;
    dVydt = diff(-P(2:end-1,:) + tauyy(2:end-1,:), 1, 2)/dY / rho0 + diff(tauxy,1,1)/dX;
    Vy(2:end-1,2:end-1) = Vy(2:end-1,2:end-1) * (1 - dt * damp) + dVydt * dt;
    
    % displacements
    Ux = Ux + Vx * dt;
    Uy = Uy + Vy * dt;
    
  % POSTPROCESSING
    if mod(it, 100000) == 0
      subplot(2, 1, 1)
      pcolor(x, y, diff(Ux,1,1)/dX)
      title(it)
      shading flat
      colorbar
      axis image        % square image
      
      subplot(2, 1, 2)
      pcolor(x, y, diff(Uy,1,2)/dY)
      title(it)
      shading flat
      colorbar
      axis image        % square image
      
      drawnow
    endif
  endfor

  S = [0 0 0];
  S(1) = mean(tauxx(:) - P(:))
  S(2) = mean(tauyy(:) - P(:))
  S(3) = mean(tauxy(:))
endfunction