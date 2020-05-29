clear, figure(1), clf

% PHYSICS
Lx  = 10.0;
K   = 1.0;
rho = 1.0;

% NUMERICS
Nx  = 100;
Nt  = 100;
CFL = 1.0;

% PREPROCESSING
dX = Lx / (Nx - 1);
x  = (-Lx / 2) : dX : (Lx / 2);
dt = CFL * dX / sqrt(K / rho);

% INITIAL CONDITIONS
P  = exp(-x .* x);
P0 = P;
V  = zeros(1,Nx+1);

% ACTION LOOP
for it = 1:Nt
  dPdt       = -diff(V) / dX * K;
  P          = P + dPdt * dt;
  dVdt       = - diff(P) / dX / rho;
  V(2:end-1) = V(2:end-1) + dVdt * dt;
% POSTPROCESSING
  if mod(it, 2) == 0
    plot(x, P0, 'g', x, P, 'r'), title(it), drawnow
  end
end
