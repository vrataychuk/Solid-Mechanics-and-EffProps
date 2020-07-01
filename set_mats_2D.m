function [K, G] = set_mats_2D(Nx, Ny, x, y)
  K0 = 1.0;
  G0 = 0.5;
  E0 = 9.0 * K0 * G0 / (3.0 * K0 + G0)
  nu0 = 0.5 * (3.0 * K0 - 2.0 * G0) / (3.0 * K0 + G0)
  K = K0 * ones(Nx, Ny);
  G = G0 * ones(Nx, Ny);
  K(sqrt(x.*x + y.*y) < 1.0) = 0.01 * K0;
  G(sqrt(x.*x + y.*y) < 1.0) = 0.01 * G0;
endfunction
