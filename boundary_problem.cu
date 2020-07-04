// Wave 2D 
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <chrono>
#include "cuda.h"



//numetric parametrs
#define CFL 0.5                //Courant-Friedrichs-Lewy
#define NGRID 2
#define NT    1000000          //number of time steps

//untouchable parametr!
#define NPARS 7                //tnks matlab for this param

__global__ void ComputeDisp(double* Ux, double* Uy, double* Vx, double* Vy, 
                            const double* const P,
                            const double* const tauXX, const double* const tauYY, const double* const tauXY,
                            const double* const pa,
                            const long int nX, const long int nY) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double dX = pa[0], dY = pa[1];
  const double dT = pa[2];
  const double rho = pa[5];
  const double damp = pa[6];

  // motion equation
  if (i > 0 && i < nX && j > 0 && j < nY - 1) {
    Vx[j * (nX + 1) + i] = Vx[j * (nX + 1) + i] * (1.0 - dT * damp) + (dT / rho) * ( (
                           -P[j * nX + i] + P[j * nX + i - 1] + tauXX[j * nX + i] - tauXX[j * nX + i - 1]
                           ) / dX + (
                           tauXY[j * (nX - 1) + i - 1] - tauXY[(j - 1) * (nX - 1) + i - 1]
                           ) / dY );
  }
  if (i > 0 && i < nX - 1 && j > 0 && j < nY) {
    Vy[j * nX + i] = Vy[j * nX + i] * (1.0 - dT * damp) + (dT / rho) * ( (
                     -P[j * nX + i] + P[(j - 1) * nX + i] + tauYY[j * nX + i] - tauYY[(j - 1) * nX + i]
                     ) / dY + (
                     tauXY[(j - 1) * (nX - 1) + i] - tauXY[(j - 1) * (nX - 1) + i - 1]
                     ) / dX );
  }

  Ux[j * (nX + 1) + i] = Ux[j * (nX + 1) + i] + Vx[j * (nX + 1) + i] * dT;
  Uy[j * nX + i] = Uy[j * nX + i] + Vy[j * nX + i] * dT;
}

__global__ void ComputeStress(const double* const Ux, const double* const Uy,
                              const double* const K, const double* const G,
                              const double* const P0, double* P,
                              double* tauXX, double* tauYY, double* tauXY,
                              const double* const pa,
                              const long int nX, const long int nY) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double dX = pa[0], dY = pa[1];
  //const double dT = pa[2];
  //const double K = pa[3], G = pa[4];

  // constitutive equation - Hooke's law
  P[j * nX + i] = P0[j * nX + i] - K[j * nX + i] * ( 
                  (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY    // divU
                  );

  tauXX[j * nX + i] = 2.0 * G[j * nX + i] * (
                      (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX -    // dUx/dx
                      ( (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY ) / 3.0    // divU / 3.0
                      );
  tauYY[j * nX + i] = 2.0 * G[j * nX + i] * (
                      (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY -    // dUy/dy
                      ( (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY ) / 3.0    // divU / 3.0
                      );

  if (i < nX - 1 && j < nY - 1) {
    tauXY[j * (nX - 1) + i] = 0.25 * (G[j * nX + i] + G[j * nX + i + 1] + G[(j + 1) * nX + i] + G[(j + 1) * nX + i + 1]) * (
                              (Ux[(j + 1) * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i + 1]) / dY + (Uy[(j + 1) * nX + i + 1] - Uy[(j + 1) * nX + i]) / dX    // dUx/dy + dUy/dx
                              );
  }
}

void SetMatrixZero(double** A_cpu, double** A_cuda, const int m, const int n) {
  *A_cpu = (double*)malloc(m * n * sizeof(double));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      (*A_cpu)[j * m + i] = 0.0;
    }
  }
  cudaMalloc(A_cuda, m * n * sizeof(double));
  cudaMemcpy(*A_cuda, *A_cpu, m * n * sizeof(double), cudaMemcpyHostToDevice);
}

void SaveMatrix(double* const A_cpu, const double* const A_cuda, const int m, const int n, const std::string filename) {
  cudaMemcpy(A_cpu, A_cuda, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  FILE* A_filw = fopen(filename.c_str(), "wb");
  fwrite(A_cpu, sizeof(double), m * n, A_filw);
  fclose(A_filw);
}

void SetMaterials(double* const K, double* const G, const int m, const int n, const double dX, const double dY, const double E0, const double nu0) {
  constexpr double E1 = 2.0;
  constexpr double nu1 = 0.2;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      K[j * m + i] = E0 / (3.0 - 6.0 * nu0);
      G[j * m + i] = E0 / (2.0 + 2.0 * nu0);
      if ( sqrt((-0.5 * dX * (m - 1) + dX * i) * (-0.5 * dX * (m - 1) + dX * i) + (-0.5 * dY * (n - 1) + dY * j) * (-0.5 * dY * (n - 1) + dY * j)) < 2.85459861019 ) {
        K[j * m + i] = E1 / (3.0 - 6.0 * nu1);
        G[j * m + i] = E1 / (2.0 + 2.0 * nu1);
      }
    }
  }
}

std::array<double, 3> ComputeSigma(const double loadValue, const std::array<int, 3>& loadType) {
  dim3 grid, block;
  block.x = 32;
  block.y = 32; 
  grid.x = NGRID;
  grid.y = NGRID;
  //physics parametrs
  const double Lx = 10;                  //physical length
  const double Ly = 10;                  //physical width
  const double E0 = 1;                   //Young's modulus
  const double nu0 = 0.25;               //Poisson's ratio
  const double rho = 1;                  //density
  // preprocessing
  const double K0 = E0 / (3 * (1 - 2 * nu0));    //bulk modulus
  const double G0 = E0 / (2 + 2 * nu0);          //shear modulus
  const long int nX = block.x * grid.x;
  const long int nY = block.y * grid.y;
  const double dX = Lx/(nX-1);
  const double dY = Ly/(nY-1);
  const double dt = CFL * min(dX, dY) / sqrt( (K0 + 4*G0/3) / rho); // time step
  const double damp = 4 / dt / nX;

  cudaSetDevice(0);
  cudaDeviceReset();
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  /* INPUT DATA */
  // parameters
  double* pa_cuda;
  double* pa_cpu = (double*)malloc(NPARS * sizeof(double));
    pa_cpu[0] = dX;
    pa_cpu[1] = dY;
    pa_cpu[2] = dt;
    pa_cpu[3] = K0;
    pa_cpu[4] = G0;
    pa_cpu[5] = rho;
    pa_cpu[6] = damp;

  cudaMalloc((void**)&pa_cuda, NPARS * sizeof(double));
  cudaMemcpy(pa_cuda, pa_cpu, NPARS * sizeof(double), cudaMemcpyHostToDevice);
  
  // materials
  double* K_cpu = (double*)malloc(nX * nY * sizeof(double));
  double* G_cpu = (double*)malloc(nX * nY * sizeof(double));
  SetMaterials(K_cpu, G_cpu, nX, nY, dX, dY, E0, nu0);
  double* K_cuda;
  double* G_cuda;
  cudaMalloc(&K_cuda, nX * nY * sizeof(double));
  cudaMalloc(&G_cuda, nX * nY * sizeof(double));
  cudaMemcpy(K_cuda, K_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(G_cuda, G_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);

  // stress
  double* P0_cuda;
  double* P0_cpu;
  SetMatrixZero(&P0_cpu, &P0_cuda, nX, nY);

  double* P_cuda;
  double* P_cpu;
  SetMatrixZero(&P_cpu, &P_cuda, nX, nY);

  double* tauXX_cuda;
  double* tauXX_cpu;
  SetMatrixZero(&tauXX_cpu, &tauXX_cuda, nX, nY);

  double* tauYY_cuda;
  double* tauYY_cpu;
  SetMatrixZero(&tauYY_cpu, &tauYY_cuda, nX, nY);

  double* tauXY_cuda;
  double* tauXY_cpu;
  SetMatrixZero(&tauXY_cpu, &tauXY_cuda, nX - 1, nY - 1);

  // displacement
  const double dUxdx = loadValue * loadType[0];
  const double dUydy = loadValue * loadType[1];
  const double dUxdy = loadValue * loadType[2];

  double* Ux_cuda;
  double* Ux_cpu = (double*)malloc((nX+1) * nY * sizeof(double));
  for (int i = 0; i < nX + 1; i++) {
    for (int j = 0; j < nY; j++) {
      Ux_cpu[j * (nX + 1) + i] = (-0.5 * dX * nX + dX * i) * dUxdx + (-0.5 * dY * (nY - 1) + dY * j) * dUxdy;
    }
  }
  cudaMalloc(&Ux_cuda, (nX + 1) * nY * sizeof(double));
  cudaMemcpy(Ux_cuda, Ux_cpu, (nX + 1) * nY * sizeof(double), cudaMemcpyHostToDevice);

  double* Uy_cuda;
  double* Uy_cpu = (double*)malloc(nX * (nY + 1) * sizeof(double));
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY + 1; j++) {
      Uy_cpu[j * nX + i] = (-0.5 * dY * nY + dY * j) * dUydy;
    }
  }
  cudaMalloc(&Uy_cuda, nX * (nY + 1) * sizeof(double));
  cudaMemcpy(Uy_cuda, Uy_cpu, nX * (nY + 1) * sizeof(double), cudaMemcpyHostToDevice);

  // velocity
  double* Vx_cuda;
  double* Vx_cpu;
  SetMatrixZero(&Vx_cpu, &Vx_cuda, nX + 1, nY);

  double* Vy_cuda;
  double* Vy_cpu;
  SetMatrixZero(&Vy_cpu, &Vy_cuda, nX, nY + 1);

  //std::cout << "Before loop...\n";

  /* ACTION LOOP */
  for (int it = 0; it < NT; it++) {
    ComputeStress<<<grid, block>>>(Ux_cuda, Uy_cuda, K_cuda, G_cuda, P0_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
    cudaDeviceSynchronize();    // wait for compute device to finish
    //std::cout << "After computing sigma...\n";
    ComputeDisp<<<grid, block>>>(Ux_cuda, Uy_cuda, Vx_cuda, Vy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
    cudaDeviceSynchronize();    // wait for compute device to finish

    /*cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Vx on step " << it << " is " << Vx_cpu[nY/2 * (nX + 1) + nX/2] << std::endl;*/
  }

  /* OUTPUT DATA WRITING */
  SaveMatrix(P_cpu, P_cuda, nX, nY, "Pc.dat");
  SaveMatrix(Uy_cpu, Uy_cuda, nX, nY + 1, "Uyc.dat");
  SaveMatrix(tauXY_cpu, tauXY_cuda, nX - 1, nY - 1, "tauXYc.dat");

  /* AVERAGING */
  cudaMemcpy(P_cpu, P_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tauXX_cpu, tauXX_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tauYY_cpu, tauYY_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tauXY_cpu, tauXY_cuda, (nX - 1) * (nY - 1) * sizeof(double), cudaMemcpyDeviceToHost);

  std::array<double, 3> Sigma = {0.0, 0.0, 0.0};
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      Sigma[0] += tauXX_cpu[j * nX + i] - P_cpu[j * nX + i];
      Sigma[1] += tauYY_cpu[j * nX + i] - P_cpu[j * nX + i];
    }
  }
  Sigma[0] /= nX * nY;
  Sigma[1] /= nX * nY;

  for (int i = 0; i < nX - 1; i++) {
    for (int j = 0; j < nY - 1; j++) {
      Sigma[2] += tauXY_cpu[j * (nX - 1) + i];
    }
  }
  Sigma[2] /= (nX - 1) * (nY - 1);

  //std::cout << Sigma[0] << '\n' << Sigma[1] << '\n' << Sigma[2] << std::endl;

  free(pa_cpu);
  free(K_cpu);
  free(G_cpu);
  free(P0_cpu);
  free(P_cpu);
  free(tauXX_cpu);
  free(tauYY_cpu);
  free(tauXY_cpu);
  free(Ux_cpu);
  free(Uy_cpu);
  free(Vx_cpu);
  free(Vy_cpu);

  cudaFree(pa_cuda);
  cudaFree(K_cuda);
  cudaFree(G_cuda);
  cudaFree(P0_cuda);
  cudaFree(P_cuda);
  cudaFree(tauXX_cuda);
  cudaFree(tauYY_cuda);
  cudaFree(tauXY_cuda);
  cudaFree(Ux_cuda);
  cudaFree(Uy_cuda);
  cudaFree(Vx_cuda);
  cudaFree(Vy_cuda);

  cudaDeviceReset();
  return Sigma;
}

int main() {
  const auto start = std::chrono::system_clock::now();

  constexpr double load_value = 0.002;
  const std::array<double, 3> Sxx = ComputeSigma(load_value, {1, 0, 0});
  const std::array<double, 3> Syy = ComputeSigma(load_value, {0, 1, 0});
  const std::array<double, 3> Sxy = ComputeSigma(load_value, {0, 0, 1});

  const double C_1111 = Sxx[0] / load_value;
  const double C_1122 = Sxx[1] / load_value;
  const double C_1112 = Sxx[2] / load_value;

  const double C_2222 = Syy[1] / load_value;
  const double C_1222 = Syy[2] / load_value;

  const double C_1212 = Sxy[2] / load_value;

  std::cout << "C_1111 = " << C_1111 << '\n';
  std::cout << "C_1122 = " << C_1122 << '\n';
  std::cout << "C_1112 = " << C_1112 << '\n';
  std::cout << "C_2222 = " << C_2222 << '\n';
  std::cout << "C_1222 = " << C_1222 << '\n';
  std::cout << "C_1212 = " << C_1212 << '\n';

  const auto end = std::chrono::system_clock::now();
  const int elapsed_sec = static_cast<int>( std::chrono::duration_cast<std::chrono::seconds>(end - start).count() );
  if (elapsed_sec < 60) {
    std::cout << "Calculation time is " << elapsed_sec << " sec\n";
  }
  else {
    const int elapsed_min = elapsed_sec / 60;
    if (elapsed_min < 60) {
      std::cout << "Calculation time is " << elapsed_min << " min " << elapsed_sec % 60 << " sec\n";
    }
    else {
      std::cout << "Calculation time is " << elapsed_min / 60 << " hours " << elapsed_min % 60 << " min " << elapsed_sec % 60 << " sec\n";
    }
  }

  return 0;
}
