function [E, nu] = set_mats_2D(Nx, Ny, x, y)
  E0 = 1.0;
  nu0 = 0.25;
  E = E0 * ones(Nx, Ny);
  nu = nu0 * ones(Nx, Ny);
  %E(sqrt(x.*x + y.*y) < 2.85459861019) = 2.0;
  %nu(sqrt(x.*x + y.*y) < 2.85459861019) = 0.2;
endfunction
