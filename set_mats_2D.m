function [E, nu] = set_mats_2D(Nx, Ny)
  E0 = 1.0;
  nu0 = 0.25;
  E = E0 * ones(Nx, Ny);
  nu = nu0 * ones(Nx, Ny);
endfunction
