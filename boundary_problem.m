clear, figure(1), clf

% PHYSICS
Lx  = 10.0;
tau = 1.0;

% NUMERICS
Nx = 100;
dt = tau / 10;

% PREPROCESSING
dX = Lx / (Nx - 1);
x  = (-Lx / 2) : dX : (Lx / 2);

% INITIAL CONDITIONS
P  = exp(-x .* x);
P0 = P;

% ACTION LOOP
for it = 1:10
  dPdt = -P / tau;
  P    = P + dPdt * dt;
% POSTPROCESSING
  plot(x, P0, 'g', x, P, 'r'), title(it), drawnow
end
