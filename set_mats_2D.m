function [K, G] = set_mats_2D(Nx, Ny, x, y)
  K0 = 1.0;
  G0 = 0.25;
  K = K0 * ones(Nx, Ny);
  G = G0 * ones(Nx, Ny);
  K(sqrt(x.*x + y.*y) < 1.0) = 0.01 * K0;
  G(sqrt(x.*x + y.*y) < 1.0) = 0.01 * G0;
endfunction
