// Wave 2D 
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include "cuda.h"

#define NGRID  1
#define NPARS  6
#define NT  10

__global__ void ComputeV(double* Vx, double* Vy, 
                         double* P,
                         double* tauXX, double* tauYY, double* tauXY,
                         double* pa,
                         const long int nX, const long int nY) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double dX = pa[0], dY = pa[1];
  const double dT = pa[2];
  const double /*K = pa[3], G = pa[4],*/ rho = pa[5];

  // motion equation
  if (i > 0 && i < nX && j > 0 && j < nY - 1) {
    Vx[j * (nX + 1) + i] = Vx[j * (nX + 1) + i] + (dT / rho) * ( (
                           -P[j * nX + i] + P[j * nX + i - 1] + tauXX[j * nX + i] - tauXX[j * nX + i - 1]
                           ) / dX + (
                           tauXY[j * (nX - 1) + i - 1] - tauXY[(j - 1) * (nX - 1) + i - 1]
                           ) / dY );
  }
  if (i > 0 && i < nX - 1 && j > 0 && j < nY) {
    Vy[j * nX + i] = Vy[j * nX + i] + (dT / rho) * ( (
                     -P[j * nX + i] + P[(j - 1) * nX + i] + tauYY[j * nX + i] - tauYY[(j - 1) * nX + i]
                     ) / dY + (
                     tauXY[(j - 1) * (nX - 1) + i] - tauXY[(j - 1) * (nX - 1) + i - 1]
                     ) / dX );
  }
}

__global__ void ComputeSigma(double* Vx, double* Vy, 
                             double* P,
                             double* tauXX, double* tauYY, double* tauXY,
                             double* pa,
                             const long int nX, const long int nY) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  const double dX = pa[0], dY = pa[1];
  const double dT = pa[2];
  const double K = pa[3], G = pa[4]/*, rho = pa[5]*/;

  // constitutive equation - Hooke's law
  P[j * nX + i] = P[j * nX + i] + (-K * ( 
                  (Vx[j * (nX + 1) + i + 1] - Vx[j * (nX + 1) + i]) / dX + (Vy[(j + 1) * nX + i] - Vy[j * nX + i]) / dY    // divV
                  ) ) * dT;

  tauXX[j * nX + i] = tauXX[j * nX + i] + 2.0 * G * (
                      (Vx[j * (nX + 1) + i + 1] - Vx[j * (nX + 1) + i]) / dX -    // dVdx
                      ( (Vx[j * (nX + 1) + i + 1] - Vx[j * (nX + 1) + i]) / dX + (Vy[(j + 1) * nX + i] - Vy[j * nX + i]) / dY ) / 3.0    // divV / 3.0
                      ) * dT;
  tauYY[j * nX + i] = tauYY[j * nX + i] + 2.0 * G * (
                      (Vy[(j + 1) * nX + i] - Vy[j * nX + i]) / dY -    // dVdy
                      ( (Vx[j * (nX + 1) + i + 1] - Vx[j * (nX + 1) + i]) / dX + (Vy[(j + 1) * nX + i] - Vy[j * nX + i]) / dY ) / 3.0    // divV / 3.0
                      ) * dT;

  if (i < nX - 1 && j < nY - 1) {
    tauXY[j * (nX - 1) + i] = tauXY[j * (nX - 1) + i] + G * (
                              (Vx[(j + 1) * (nX + 1) + i + 1] - Vx[j * (nX + 1) + i + 1]) / dY + (Vy[(j + 1) * nX + i + 1] - Vy[(j + 1) * nX + i]) / dX
                              ) * dT;
  }
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
  double* P_cuda;
  double* P_cpu = (double*)malloc(nX * nY * sizeof(double));
  FILE* P_fil = fopen("P.dat", "rb");
  if (!P_fil) {
    std::cerr << "Error! Cannot open file P.dat!\n";
    return 1;
  }
  fread(P_cpu, sizeof(double), nX * nY, P_fil);
  fclose(P_fil);
  cudaMalloc(&P_cuda, nX * nY * sizeof(double));
  cudaMemcpy(P_cuda, P_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);

  double* tauXX_cuda;
  double* tauXX_cpu = (double*)malloc(nX * nY * sizeof(double));
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      tauXX_cpu[j * nX + i] = 0.0;
    }
  }
  cudaMalloc(&tauXX_cuda, nX * nY * sizeof(double));
  cudaMemcpy(tauXX_cuda, tauXX_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);

  double* tauYY_cuda;
  double* tauYY_cpu = (double*)malloc(nX * nY * sizeof(double));
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      tauYY_cpu[j * nX + i] = 0.0;
    }
  }
  cudaMalloc(&tauYY_cuda, nX * nY * sizeof(double));
  cudaMemcpy(tauYY_cuda, tauYY_cpu, nX * nY * sizeof(double), cudaMemcpyHostToDevice);

  double* tauXY_cuda;
  double* tauXY_cpu = (double*)malloc((nX - 1) * (nY - 1) * sizeof(double));
  for (int i = 0; i < nX - 1; i++) {
    for (int j = 0; j < nY - 1; j++) {
      tauXY_cpu[j * (nX - 1) + i] = 0.0;
    }
  }
  cudaMalloc(&tauXY_cuda, (nX - 1) * (nY - 1) * sizeof(double));
  cudaMemcpy(tauXY_cuda, tauXY_cpu, (nX - 1) * (nY - 1) * sizeof(double), cudaMemcpyHostToDevice);

  // velocity
  double* Vx_cuda;
  double* Vx_cpu = (double*)malloc((nX+1) * nY * sizeof(double));
  for (int i = 0; i < nX + 1; i++) {
    for (int j = 0; j < nY; j++) {
      Vx_cpu[j * (nX + 1) + i] = 0.0;
    }
  }
  cudaMalloc(&Vx_cuda, (nX + 1) * nY * sizeof(double));
  cudaMemcpy(Vx_cuda, Vx_cpu, (nX + 1) * nY * sizeof(double), cudaMemcpyHostToDevice);

  double* Vy_cuda;
  double* Vy_cpu = (double*)malloc(nX * (nY + 1) * sizeof(double));
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY + 1; j++) {
      Vy_cpu[j * nX + i] = 0.0;
    }
  }
  cudaMalloc(&Vy_cuda, nX * (nY + 1) * sizeof(double));
  cudaMemcpy(Vy_cuda, Vy_cpu, nX * (nY + 1) * sizeof(double), cudaMemcpyHostToDevice);

  //std::cout << "Before loop...\n";

  /* ACTION LOOP */
  for (int it = 0; it < NT; it++) {
    ComputeSigma<<<grid, block>>>(Vx_cuda, Vy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
    cudaDeviceSynchronize();    // wait for compute device to finish
    //std::cout << "After computing sigma...\n";
    ComputeV<<<grid, block>>>(Vx_cuda, Vy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
    cudaDeviceSynchronize();    // wait for compute device to finish

    cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Vx on step " << it << " is " << Vx_cpu[nY/2 * (nX + 1) + nX/2] << std::endl;
  }

  /* OUTPUT DATA WRITING */
  cudaMemcpy(P_cpu, P_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);

  FILE* P_filw = fopen("Pc.dat", "wb");
  fwrite(P_cpu, sizeof(double), nX * nY, P_filw);
  fclose(P_filw);

  cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost);
  FILE* Vx_filw = fopen("Vxc.dat", "wb");
  fwrite(Vx_cpu, sizeof(double), (nX + 1) * nY, Vx_filw);
  fclose(Vx_filw);

  cudaMemcpy(tauXY_cpu, tauXY_cuda, (nX - 1) * (nY - 1) * sizeof(double), cudaMemcpyDeviceToHost);
  FILE* tauXY_filw = fopen("tauXYc.dat", "wb");
  fwrite(tauXY_cpu, sizeof(double), (nX - 1) * (nY - 1), tauXY_filw);
  fclose(tauXY_filw);

  free(pa_cpu);
  free(P_cpu);
  free(tauXX_cpu);
  free(tauYY_cpu);
  free(tauXY_cpu);
  free(Vx_cpu);
  free(Vy_cpu);

  cudaFree(pa_cuda);
  cudaFree(P_cuda);
  cudaFree(tauXX_cuda);
  cudaFree(tauYY_cuda);
  cudaFree(tauXY_cuda);
  cudaFree(Vx_cuda);
  cudaFree(Vy_cuda);

  cudaDeviceReset();
  return 0;
}
