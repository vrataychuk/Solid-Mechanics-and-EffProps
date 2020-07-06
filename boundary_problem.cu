// Wave 2D 
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <chrono>
#include "cuda.h"
#include <cmath>



//numetric parametrs
#define CFL 0.125                //Courant-Friedrichs-Lewy
#define NGRID 2
#define NT    10                 //number of time steps
#define nIter 100000

//untouchable parametr!
#define NPARS 8                //tnks matlab for this param

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
  const double dampX = pa[6];
  const double dampY = pa[7];

  // motion equation
  if (i > 0 && i < nX && j > 0 && j < nY - 1) {
    Vx[j * (nX + 1) + i] = Vx[j * (nX + 1) + i] * (1.0 - dT * dampX) + (dT / rho) * ( (
                           -P[j * nX + i] + P[j * nX + i - 1] + tauXX[j * nX + i] - tauXX[j * nX + i - 1]
                           ) / dX + (
                           tauXY[j * (nX - 1) + i - 1] - tauXY[(j - 1) * (nX - 1) + i - 1]
                           ) / dY );
  }
  if (i > 0 && i < nX - 1 && j > 0 && j < nY) {
    Vy[j * nX + i] = Vy[j * nX + i] * (1.0 - dT * dampY) + (dT / rho) * ( (
                     -P[j * nX + i] + P[(j - 1) * nX + i] + tauYY[j * nX + i] - tauYY[(j - 1) * nX + i]
                     ) / dY + (
                     tauXY[(j - 1) * (nX - 1) + i] - tauXY[(j - 1) * (nX - 1) + i - 1]
                     ) / dX );
  }

  Ux[j * (nX + 1) + i] = Ux[j * (nX + 1) + i] + Vx[j * (nX + 1) + i] * dT;
  Uy[j * nX + i] = Uy[j * nX + i] + Vy[j * nX + i] * dT;
}

__global__ void ComputeTauxyAv(double* tauxyAv, const double* const tauXY, const long int nX, const long int nY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nX - 2 && j < nY - 2) {
        tauxyAv[(j + 1) * nX + (i + 1)] = 0.25 * (tauXY[j * (nX - 1) + i] + tauXY[j * (nX - 1) + i + 1] + tauXY[(j + 1) * (nX - 1) + i] + tauXY[(j + 1) * (nX - 1) + i + 1]);
    }
}

__global__ void ComputeBordTauxyAv(double* tauxyAv, const long int nX, const long int nY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i == 0 && j < nY - 2) {
        tauxyAv[(j + 1) * nX + i] = tauxyAv[(j + 1) * nX + i + 1];
    }
    if (i == (nX - 1) && j < nY - 2) {
        tauxyAv[(j + 1) * nX + i] = tauxyAv[(j + 1) * nX + i - 1];
    }
    if (j == 0 && i < nX - 2) {
        tauxyAv[j * nX + i + 1] = tauxyAv[(j + 1) * nX + i + 1];
    }
    if (j == (nY - 1) && i < nX - 2) {
        tauxyAv[j * nX + i + 1] = tauxyAv[(j - 1) * nX + i + 1];
    }
    if (i == 0 && j == 0) {
        tauxyAv[0] = 0.5 * (tauxyAv[nX] + tauxyAv[1]);
    }
    if (i == (nX - 1) && j == 0) {
        tauxyAv[i] = 0.5 * (tauxyAv[nX+i] + tauxyAv[i - 1]);
    }
    if (i == 0 && j == (nY - 1)) {
        tauxyAv[j * nX] = 0.5 * (tauxyAv[j * nX + 1] + tauxyAv[(j - 1) * nX]);
    }
    if (i == (nX - 1) && j == (nY - 1)) {
        tauxyAv[j * nX + i] = 0.5 * (tauxyAv[j * nX + i - 1] + tauxyAv[(j - 1) * nX + i]);
    }
}

__global__ void ComputeLoad(double* Ux, double* Uy, const double dUxdx, const double dUydy,
                          const double dUxdy, const double dX, const double dY, const int nt, const long int nX, const long int nY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < (nX + 1) && j < nY) {
        Ux[j * (nX + 1) + i] = Ux[j * (nX + 1) + i] + ((-0.5 * dX * nX + dX * i) * dUxdx + (-0.5 * dY * (nY - 1) + dY * j) * dUxdy) / nt;
    }
    if (i < nX && j < (nY + 1)) {
        Uy[j * nX + i] = Uy[j * nX + i] + ((-0.5 * dY * nY + dY * j) * dUydy) / nt;
    }
}

__global__ void ComputePlast(double* J2, double* J2xy, double* tauXX, double* tauYY, double* tauXY, double* tauxyAv,
                             double* Plast, double* PlastXY, const long int nX, const long int nY, const double coh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    /* TRESCA CRITERIA */
    J2[j * nX + i] = sqrt((tauXX[j * nX + i]) * (tauXX[j * nX + i]) + (tauYY[j * nX + i]) * (tauYY[j * nX + i]) + 2 * (tauxyAv[j * nX + i]) * (tauxyAv[j * nX + i]));
    
    if (J2[j * nX + i] > coh) {
        Plast[j * nX + i] = 1;
        tauXX[j * nX + i] = tauXX[j * nX + i] * coh / J2[j * nX + i];
        tauYY[j * nX + i] = tauYY[j * nX + i] * coh / J2[j * nX + i];
    }
    if (i < nX - 1 && j < nY - 1) {
        J2xy[j * (nX - 1) + i] = sqrt((0.25 * (tauXX[j * nX + i] + tauXX[j * nX + i + 1] + tauXX[(j + 1) * nX + i] + tauXX[(j + 1) * nX + i + 1]))
            * (0.25 * (tauXX[j * nX + i] + tauXX[j * nX + i + 1] + tauXX[(j + 1) * nX + i] + tauXX[(j + 1) * nX + i + 1]))
            + (0.25 * (tauYY[j * nX + i] + tauYY[j * nX + i + 1] + tauYY[(j + 1) * nX + i] + tauYY[(j + 1) * nX + i + 1]))
            * (0.25 * (tauYY[j * nX + i] + tauYY[j * nX + i + 1] + tauYY[(j + 1) * nX + i] + tauYY[(j + 1) * nX + i + 1]))
            + 2 * tauXY[j * (nX - 1) + i] * tauXY[j * (nX - 1) + i]);
        if (J2xy[j * (nX - 1) + i] > coh) {
            PlastXY[j * (nX - 1) + i] = 1;
            tauXY[j * (nX - 1) + i] = tauXY[j * (nX - 1) + i] * coh / J2xy[j * (nX - 1) + i];
        }
    }
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

void SetMaterials(double* const K, double* const G, const int m, const int n, const double dX, const double dY, const double rad, const double K0, const double G0, const double EPC) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      K[j * m + i] = K0;
      G[j * m + i] = G0;
      if ( sqrt((-0.5*dX * (m - 1) + dX * i)* (-0.5*dX * (m - 1) + dX * i)+(-0.5*dY*(n-1)+dY*j)* (-0.5*dY * (n - 1) + dY * j))<rad) {
          K[j * m + i] = EPC*K0;
          G[j * m + i] = EPC*G0;
      }
    }
  }
}

std::array<double, 3> ComputeSigma(const double loadValue, const std::array<double, 3>& loadType) {
  dim3 grid, block;
  block.x = 32;
  block.y = 32;
  //time parametrs
  int one = 0;
  int two = 0;
  int tree = 0;
  double nt1 = NT / 4;
  double nt2 = NT / 2;
  double nt3 = 3 * NT / 4;
  //physics parametrs
  const double Lx = 20;                  //physical length
  const double Ly = 20;                  //physical width
  const double E0 = 1;                   //Young's modulus
  const double nu0 = 0.25;               //Poisson's ratio
  const double rho = 1;                  //density
  const double coh = 0.01;
  const double P0 = 1.0 * coh;
  const double rad = 1.0;                //rad of the hole
  const double EPC = 0.01;               //eff props comperesent of the hole and surface
  // preprocessing
  grid.x = NGRID;
  grid.y = NGRID;
  const double K0 = E0 / (3 * (1 - 2 * nu0));    //bulk modulus
  const double G0 = E0 / (2 + 2 * nu0);          //shear modulus
  const long int nX = block.x * grid.x;
  const long int nY = block.y * grid.y;
  const double dX = Lx/(nX-1);
  const double dY = Ly/(nY-1);
  const double dt = CFL * min(dX, dY) / sqrt( (K0 + 4*G0/3) / rho); // time step
  const double dampX = 4 / dt / nX;
  const double dampY = 4 / dt / nY;

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
    pa_cpu[6] = dampX;
    pa_cpu[7] = dampY;

  cudaMalloc((void**)&pa_cuda, NPARS * sizeof(double));
  cudaMemcpy(pa_cuda, pa_cpu, NPARS * sizeof(double), cudaMemcpyHostToDevice);
  
  // materials
  double* K_cpu = (double*)malloc(nX * nY * sizeof(double));
  double* G_cpu = (double*)malloc(nX * nY * sizeof(double));
  SetMaterials(K_cpu, G_cpu, nX, nY, dX, dY, rad, K0, G0, EPC);
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

  double* tauxyAv_cuda;
  double* tauxyAv_cpu;
  SetMatrixZero(&tauxyAv_cpu, &tauxyAv_cuda, nX, nY);

  //for plasticity
  double* J2_cuda;
  double* J2_cpu;
  SetMatrixZero(&J2_cpu, &J2_cuda, nX, nY);

  double* J2xy_cuda;
  double* J2xy_cpu;
  SetMatrixZero(&J2xy_cpu, &J2xy_cuda, nX - 1, nY - 1);

  double* Plast_cuda;
  double* Plast_cpu;
  SetMatrixZero(&Plast_cpu, &Plast_cuda, nX, nY);

  double* PlastXY_cuda;
  double* PlastXY_cpu;
  SetMatrixZero(&PlastXY_cpu, &PlastXY_cuda, nX - 1, nY - 1);

  // displacement
  const double dUxdx = loadValue * loadType[0];
  const double dUydy = loadValue * loadType[1];
  const double dUxdy = loadValue * loadType[2];

  double* Ux_cuda;
  double* Ux_cpu;
  SetMatrixZero(&Ux_cpu, &Ux_cuda, nX + 1, nY);

  double* Uy_cuda;
  double* Uy_cpu;
  SetMatrixZero(&Uy_cpu, &Uy_cuda, nX, nY + 1);

  // velocity
  double* Vx_cuda;
  double* Vx_cpu;
  SetMatrixZero(&Vx_cpu, &Vx_cuda, nX + 1, nY);

  double* Vy_cuda;
  double* Vy_cpu;
  SetMatrixZero(&Vy_cpu, &Vy_cuda, nX, nY + 1);

  //for effprops
  double deltaP;
  double deltaP1;
  double deltaP2;
  double tauInfty;
  double tauInfty1;
  double tauInfty2;
  double divUeff = loadValue * (loadType[0]+loadType[1]);
  double Keff;
  double meanP;
  double meantauXX;
  double meantauYY;
  double Geff1;
  double Geff2;

  //std::cout << "Before loop...\n";

  /* ACTION LOOP */
  for (int it = 0; it < NT; it++) {
      //set zero
      deltaP = 0;
      deltaP1 = 0;
      deltaP2 = 0;
      tauInfty1 = 0;
      tauInfty2 = 0;
      tauInfty = 0;
      meanP = 0;
      meantauXX = 0;
      meantauYY = 0;
      ComputeLoad<<<grid, block>>>(Ux_cuda, Uy_cuda, dUxdx, dUydy, dUxdy, dX, dY, NT, nX, nY);
      cudaDeviceSynchronize();    // wait for compute device to finish
      for (int iter = 0; iter < nIter;iter++) {
          ComputeStress<<<grid, block>>>(Ux_cuda, Uy_cuda, K_cuda, G_cuda, P0_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
          cudaDeviceSynchronize();    // wait for compute device to finish
          //tauxy for plasticiti
          ComputeTauxyAv<<<grid, block>>>(tauxyAv_cuda, tauXY_cuda, nX, nY);      //tauxyAv(2:end-1,2:end-1) = av4(tauxy);
          cudaDeviceSynchronize();    // wait for compute device to finish
          ComputeBordTauxyAv<<<grid, block>>>(tauxyAv_cuda, nX, nY);
          cudaDeviceSynchronize();    // wait for compute device to finish
          //plasticiti
          ComputePlast<<<grid, block>>>(J2_cuda, J2xy_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, tauxyAv_cuda, Plast_cuda, PlastXY_cuda, nX, nY, coh);
          cudaDeviceSynchronize();    // wait for compute device to finish
          ComputeDisp<<<grid, block>>>(Ux_cuda, Uy_cuda, Vx_cuda, Vy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, pa_cuda, nX, nY);
          cudaDeviceSynchronize();    // wait for compute device to finish
      }
      if ((it > nt1) && (one == 0)) {
          std::cout << "25%" << '\n';
          one = 1;
      }
      else if ((it > nt2) && (two == 0)) {
          std::cout << "50%" << '\n';
          two = 1;
      }
      else if ((it > nt3) && (tree == 0)) {
          std::cout << "75%" << '\n';
          tree = 1;
      }
      cudaMemcpy(P_cpu, P_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(tauXX_cpu, tauXX_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(tauYY_cpu, tauYY_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost);
      for (int j = 0; j < nY; j++) {
          deltaP1 = deltaP1 + tauXX_cpu[j * nX] + tauYY_cpu[j * nX] - 2 * P_cpu[j * nX];
          deltaP1 = deltaP1 + tauXX_cpu[j * nX + nX - 1] + tauYY_cpu[j * nX + nX - 1] - 2 * P_cpu[j * nX + nX - 1];
          tauInfty1 = tauInfty1 + tauXX_cpu[j * nX] - tauYY_cpu[j * nX] + tauXX_cpu[j * nX + nX - 1] - tauYY_cpu[j * nX + nX - 1];
      }
      deltaP1 = deltaP1 / nY;
      tauInfty1 = tauInfty1 / nY;
      for (int i = 0; i < nX; i++) {
          deltaP2 = deltaP2 + tauXX_cpu[i] + tauYY_cpu[i] - 2 * P_cpu[i];
          deltaP2 = deltaP2 + tauXX_cpu[(nY - 1) * nX + i] + tauYY_cpu[(nY - 1) * nX + i] - 2 * P_cpu[(nY - 1) * nX + i];
          tauInfty2 = tauInfty2 + tauXX_cpu[i] - tauYY_cpu[i] + tauXX_cpu[(nY - 1) * nX + i] - tauYY_cpu[(nY - 1) * nX + i];
          for (int j = 0;j < nY;j++) {
              meanP = meanP + P_cpu[j * nX + i];
              meantauXX = meantauXX + tauXX_cpu[j * nX + i];
              meantauYY = meantauYY + tauYY_cpu[j * nX + i];
          }
      }
      meanP = meanP / nX / nY;
      meantauXX = meantauXX / nX / nY;
      meantauYY = meantauYY / nX / nY;
      deltaP2 = deltaP2 / nX;
      tauInfty2 = tauInfty2 / nX;
      deltaP = (deltaP1 + deltaP2) * (CFL) / (coh * sqrt(2));
      tauInfty = (tauInfty1 + tauInfty2) * (CFL) / (coh * sqrt(2));
      Keff = -(meanP * NT) / (divUeff * (it + 1));
      Geff1 = (0.5 * meantauXX * NT) / ((it + 1) * (loadValue * loadType[0] - divUeff / 3));
      Geff2 = (0.5 * meantauYY * NT) / ((it + 1) * (loadValue * loadType[1] - divUeff / 3));
      std::cout << "it =   " << it << "\n deltaP =   " << deltaP << "\n tauInfty =   " << tauInfty << "\n Keff =   " <<
          Keff << "\n Geff1 =   " << Geff1 << "\n Geff2 =   " << Geff2<<'\n';
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
  free(tauxyAv_cpu);
  free(J2_cpu);
  free(J2xy_cpu);
  free(Plast_cpu);
  free(PlastXY_cpu);

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
  cudaFree(tauxyAv_cuda);
  cudaFree(J2_cuda);
  cudaFree(J2xy_cuda);
  cudaFree(Plast_cuda);
  cudaFree(PlastXY_cuda);

  cudaDeviceReset();
  return Sigma;
}

int main() {
  const auto start = std::chrono::system_clock::now();

  constexpr double load_value = 0.002;
  
  const std::array<double, 3> Sxx = ComputeSigma(load_value, {1, 1, 0});
  
  /*const std::array<double, 3> Syy = ComputeSigma(load_value, {0, 1, 0});
  const std::array<double, 3> Sxy = ComputeSigma(load_value, {0, 0, 1});*/

  /*const double C_1111 = Sxx[0] / load_value;
  const double C_1122 = Sxx[1] / load_value;
  const double C_1112 = Sxx[2] / load_value;

  const double C_2222 = Syy[1] / load_value;
  const double C_1222 = Syy[2] / load_value;

  const double C_1212 = Sxy[2] / load_value;*/

  /*std::cout << "C_1111 = " << C_1111 << '\n';
  std::cout << "C_1122 = " << C_1122 << '\n';
  std::cout << "C_1112 = " << C_1112 << '\n';
  std::cout << "C_2222 = " << C_2222 << '\n';
  std::cout << "C_1222 = " << C_1222 << '\n';
  std::cout << "C_1212 = " << C_1212 << '\n';*/

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
