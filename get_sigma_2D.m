function S = get_sigma_2D(loadValue, loadType, Nt)
  figure(1)
  clf
  colormap jet

  % PHYSICS
  Lx  = 10.0;                         % physical length
  Ly  = 10.0;                         % physical width
  E0   = 1.0;                         % Young's modulus
  nu0  = 0.25;                        % Poisson's ratio  
  rho0 = 1.0;                         % density
  K0   = E0 / (3.0 * (1 - 2 * nu0));  % bulk modulus
  G0   = E0 / (2.0 + 2.0 * nu0);      % shear modulus
  coh  = 0.00075;

  % NUMERICS
  Nx  = 100;     % number of space steps
  Ny  = 100;
  %Nt  = 10;      % number of time steps
  nIter = 100;
  CFL = 0.5;     % Courant–Friedrichs–Lewy

  % PREPROCESSING
  dX     = Lx / (Nx - 1);                                   % space step
  dY     = Ly / (Ny - 1);
  x      = (-Lx / 2) : dX : (Lx / 2);                       % space discretization
  y      = (-Ly / 2) : dY : (Ly / 2);
  [x, y] = ndgrid(x, y);                                    % 2D mesh
  [xUx, yUx] = ndgrid((-(Lx + dX)/2) : dX : ((Lx + dX)/2), (-Ly/2) : dY : (Ly/2));
  [xUy, yUy] = ndgrid((-Lx/2) : dX : (Lx/2), (-(Ly+dY)/2) : dY : ((Ly+dY)/2));
  dt     = CFL * min(dX, dY) / sqrt( (K0 + 4*G0/3) / rho0);    % time step
  dampX   = 4.0 / dt / Nx;
  dampY   = 4.0 / dt / Ny;
  
  % MATERIALS
  E = zeros(Nx, Ny);
  nu = zeros(Nx, Ny);
  [E, nu] = set_mats_2D(Nx, Ny, x, y);     % Young's modulus and Poisson's ratio
  K = E ./ (3.0 * (1 - 2 * nu));             % bulk modulus
  G = E ./ (2.0 + 2.0 * nu);                 % shear modulus

  % INITIAL CONDITIONS
  P0    = zeros(Nx, Ny);            % initial hydrostatic stress
  tauxyAv = zeros(Nx, Ny);
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
  
  S = zeros(Nt, 3);

  % ACTION LOOP
  for it = 1 : Nt
    Ux = Ux + (dUxdx * xUx + dUxdy * yUx) / Nt;
    Uy = Uy + (dUydy * yUy) / Nt;
    for iter = 1 : nIter
      % displacement divergence
      divU = diff(Ux,1,1) / dX + diff(Uy,1,2) / dY;
      
      % constitutive equation - Hooke's law
      P     = P0 - K .* divU;
      tauxx = 2.0 * G .* (diff(Ux,1,1)/dX - divU/3.0);
      tauyy = 2.0 * G .* (diff(Uy,1,2)/dY - divU/3.0);
      tauxy = av4(G) .* (diff(Ux(2:end-1,:), 1, 2)/dY + diff(Uy(:,2:end-1), 1, 1)/dX);
      
      % tauXY for plasticity
      tauxyAv(2:end-1,2:end-1) = av4(tauxy);
      
      tauxyAv(1, 2:end-1) = tauxyAv(2, 2:end-1);
      tauxyAv(end, 2:end-1) = tauxyAv(end-1, 2:end-1);
      tauxyAv(2:end-1, 1) = tauxyAv(2:end-1, 2);
      tauxyAv(2:end-1, end) = tauxyAv(2:end-1, end-1);
      tauxyAv(1, 1) = 0.5 * (tauxyAv(1, 2) + tauxyAv(2, 1));
      tauxyAv(end, 1) = 0.5 * (tauxyAv(end, 2) + tauxyAv(end-1, 1));
      tauxyAv(1, end) = 0.5 * (tauxyAv(2, end) + tauxyAv(1, end-1));
      tauxyAv(end, end) = 0.5 * (tauxyAv(end, end-1) + tauxyAv(end-1, end));
      
      % plasticity
      J2 = sqrt(tauxx .* tauxx + tauyy .* tauyy + 2.0 * tauxyAv .* tauxyAv);    % Tresca criteria
      J2xy = sqrt(av4(tauxx).^2 + av4(tauyy).^2 + 2.0 * tauxy .* tauxy);
      iPlast = find(J2 > coh);
      if length(iPlast) > 0
        tauxx(iPlast) = tauxx(iPlast) .* coh ./ J2(iPlast);
        tauyy(iPlast) = tauyy(iPlast) .* coh ./ J2(iPlast);
      endif
      iPlastXY = find(J2xy > coh);
      if length(iPlastXY) > 0
        tauxy(iPlastXY) = tauxy(iPlastXY) .* coh ./ J2xy(iPlastXY);
      endif
      
      % motion equation
      dVxdt = diff(-P(:,2:end-1) + tauxx(:,2:end-1), 1, 1)/dX / rho0 + diff(tauxy,1,2)/dY;
      Vx(2:end-1,2:end-1) = Vx(2:end-1,2:end-1) * (1 - dt * dampX) + dVxdt * dt;
      dVydt = diff(-P(2:end-1,:) + tauyy(2:end-1,:), 1, 2)/dY / rho0 + diff(tauxy,1,1)/dX;
      Vy(2:end-1,2:end-1) = Vy(2:end-1,2:end-1) * (1 - dt * dampY) + dVydt * dt;
      
      % displacements
      Ux = Ux + Vx * dt;
      Uy = Uy + Vy * dt;
    endfor
    
  % POSTPROCESSING
    if mod(it, 2) == 0
      %subplot(2, 1, 1)
      %pcolor(x, y, diff(Ux,1,1)/dX)
      %title(it)
      %shading flat
      %colorbar
      %axis image        % square image
      
      %subplot(2, 1, 2)
      %pcolor(x, y, diff(Uy,1,2)/dY)
      %title(it)
      %shading flat
      %colorbar
      %axis image        % square image
      
      %drawnow
    endif
    
    S(it, 1) = mean(tauxx(:) - P(:));
    S(it, 2) = mean(tauyy(:) - P(:));
    S(it, 3) = mean(tauxy(:));
  endfor
  
endfunction
