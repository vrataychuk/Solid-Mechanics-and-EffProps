// Wave 2D 
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include "cuda.h"

#define NGRID 2
#define NPARS 7
#define NT    10000

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
                              const double* const P0, double* P,
                              double* tauXX, double* tauYY, double* tauXY,
                              const double* const pa,
                              const long int nX, const long int nY) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double dX = pa[0], dY = pa[1];
  //const double dT = pa[2];
  const double K = pa[3], G = pa[4];

  // constitutive equation - Hooke's law
  P[j * nX + i] = P0[j * nX + i] - K * ( 
                  (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY    // divU
                  );

  tauXX[j * nX + i] = 2.0 * G * (
                      (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX -    // dUx/dx
                      ( (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY ) / 3.0    // divU / 3.0
                      );
  tauYY[j * nX + i] = 2.0 * G * (
                      (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY -    // dUy/dy
                      ( (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY ) / 3.0    // divU / 3.0
                      );

  if (i < nX - 1 && j < nY - 1) {
    tauXY[j * (nX - 1) + i] = G * (
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

int main() {
  dim3 grid, block;
  block.x = 32; 
  block.y = 32; 
  grid.x = NGRID;
  grid.y = NGRID;

  const long int nX = block.x * grid.x;
  const long int nY = block.y * grid.y;

  cudaSetDevice(0);
  cudaDeviceReset();
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  /* INPUT DATA READING */
  // parameters
  double* pa_cuda;
  double* pa_cpu = (double*)malloc(NPARS * sizeof(double));
  //std::ifstream pa_fil("pa.dat", std::ifstream::in | std::ifstream::binary);
  FILE* pa_fil = fopen("pa.dat", "rb");
  if (!pa_fil) {
    std::cerr << "Error! Cannot open file pa.dat!\n";
    return 1;
  }
  //pa_fil.read(pa_cpu, NPARS * sizeof(double));
  fread(pa_cpu, sizeof(double), NPARS, pa_fil);
  //pa_fil.close();
  fclose(pa_fil);
  cudaMalloc((void**)&pa_cuda, NPARS * sizeof(double));
  cudaMemcpy(pa_cuda, pa_cpu, NPARS * sizeof(double), cudaMemcpyHostToDevice);

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
  const double dX = pa_cpu[0], dY = pa_cpu[1];
  const double dUxdx = 0.0;
  const double dUydy = 0.002;
  const double dUxdy = 0.0;

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
    ComputeStress<<<grid, block>>>(Ux_cuda, Uy_cuda, P0_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
    cudaDeviceSynchronize();    // wait for compute device to finish
    //std::cout << "After computing sigma...\n";
    ComputeDisp<<<grid, block>>>(Ux_cuda, Uy_cuda, Vx_cuda, Vy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
    cudaDeviceSynchronize();    // wait for compute device to finish

    /*cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Vx on step " << it << " is " << Vx_cpu[nY/2 * (nX + 1) + nX/2] << std::endl;*/
  }

  /* OUTPUT DATA WRITING */
  cudaMemcpy(P_cpu, P_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
  FILE* P_filw = fopen("Pc.dat", "wb");
  fwrite(P_cpu, sizeof(double), nX * nY, P_filw);
  fclose(P_filw);

  cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  FILE* Uy_filw = fopen("Uyc.dat", "wb");
  fwrite(Uy_cpu, sizeof(double), nX * (nY + 1), Uy_filw);
  fclose(Uy_filw);

  cudaMemcpy(tauXY_cpu, tauXY_cuda, (nX - 1) * (nY - 1) * sizeof(double), cudaMemcpyDeviceToHost);
  FILE* tauXY_filw = fopen("tauXYc.dat", "wb");
  fwrite(tauXY_cpu, sizeof(double), (nX - 1) * (nY - 1), tauXY_filw);
  fclose(tauXY_filw);

  free(pa_cpu);
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
  return 0;
}
