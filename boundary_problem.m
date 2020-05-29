clear, figure(1), clf

% PHYSICS
Lx  = 10.0;    % physical length
K   = 1.0;     % bulk modulus
rho = 1.0;     % density
G   = 1.5;     % shear modulus

% NUMERICS
Nx  = 100;     % number of space steps
Nt  = 100;     % number of time steps
CFL = 1.0;     % Courant–Friedrichs–Lewy

% PREPROCESSING
dX = Lx / (Nx - 1);                          % space step
x  = (-Lx / 2) : dX : (Lx / 2);              % space discretization
dt = CFL * dX / sqrt( (K + 4*G/3) / rho);    % time step

% INITIAL CONDITIONS
P   = exp(-x .* x);        % hydrostatic stress (ball part of tensor)
P0  = P;                   % initial hydrostatic stress
V   = zeros(1, Nx + 1);    % velocity
tau = zeros(1, Nx);        % deviatoric stress

% ACTION LOOP
for it = 1 : Nt
  P          = P + (-diff(V) / dX * K)                  * dt;
  tau        = tau + (diff(V)/dX * G * 4.0 / 3.0)       * dt;
  V(2:end-1) = V(2:end-1) + (diff(-P + tau) / dX / rho) * dt;
% POSTPROCESSING
  if mod(it, 5) == 0
    plot(x, P0, 'g', x, P, 'r')
    title(it)
    drawnow
  end
end
