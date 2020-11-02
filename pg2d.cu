#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <chrono>
#include "cuda.h"
#include <cmath>

__global__ void ComputeE(double* Exx, double* Eyy, double* Exy, const double* const Vx, const double* const Vy,
                         const double dx, const double dy, const long int nx, const long int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Exx[j * nx + i] = (Vx[j * (nx + 1) + i + 1] - Vx[j * (nx + 1) + i]) / dx;
    Eyy[j * nx + i] = (Vy[(j + 1) * nx + i] - Vy[j * nx + i]) / dy;
    if (i < (nx - 1) && j < (ny - 1)) {
        Exy[j * (nx - 1) + i] = 0.5 * ((Vx[(j + 1) * (nx + 1) + i + 1] - Vx[j * (nx + 1) + i + 1]) / dy + (Vy[(j + 1) * nx + i + 1] - Vy[(j + 1) * nx + i]) / dx);
    }
}

__global__ void ComputeTau(double* P, double* tauXX, double* tauYY, double* tauXY, const double* const Exx, const double* const Eyy, const double* const Exy,
                           const double G0, const double K0, const double dt, const long int nx, const long int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    P[j * nx + i] = P[j * nx + i] - K0 * dt * (Exx[j * nx + i] + Eyy[j * nx + i]);
    tauXX[j * nx + i] = tauXX[j * nx + i] + 2 * G0 * dt * (Exx[j * nx + 1] - (Exx[j * nx + i] + Eyy[j * nx + i]) / 3);
    tauYY[j * nx + i] = tauYY[j * nx + i] + 2 * G0 * dt * (Eyy[j * nx + 1] - (Exx[j * nx + i] + Eyy[j * nx + i]) / 3);
    if (i < nx - 1 && j < ny - 1) {
        tauXY[j * (nx - 1) + i] = tauXY[j * (nx - 1) + i] + 2 * G0 * dt * Exy[j * (nx - 1) + i];
    }
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
        tauxyAv[i] = 0.5 * (tauxyAv[nX + i] + tauxyAv[i - 1]);
    }
    if (i == 0 && j == (nY - 1)) {
        tauxyAv[j * nX] = 0.5 * (tauxyAv[j * nX + 1] + tauxyAv[(j - 1) * nX]);
    }
    if (i == (nX - 1) && j == (nY - 1)) {
        tauxyAv[j * nX + i] = 0.5 * (tauxyAv[j * nX + i - 1] + tauxyAv[(j - 1) * nX + i]);
    }
}

__global__ void ComputePlast(double* J2, double* tauXX, double* tauYY, double* tauXY, double* tauxyAv,
                             double* lam, const long int nX, const long int nY, const double coh, int* flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    J2[j * nX + i] = sqrt(((tauXX[j * nX + i]) * (tauXX[j * nX + i]) + (tauYY[j * nX + i]) * (tauYY[j * nX + i]))/2 +  (tauxyAv[j * nX + i]) * (tauxyAv[j * nX + i]));
    lam[j * nX + i] = (1 - coh / J2[j * nX + i]);
    if (lam[j * nX + i] < 0) {
        lam[j * nX + i] = 0;
    }
    else if (lam[j * nX + i] > 0) {
        flag[0]+=1;
    }
}

__global__ void ComputePlastCor(double* tauXX, double* tauYY, double* tauXY, double* tauxyAV, double* J2, double* lama, double* lam, const long int nx, const long int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    tauXX[j * nx + i] = (1 - lam[j * nx + i]) * tauXX[j * nx + i];
    tauYY[j * nx + i] = (1 - lam[j * nx + i]) * tauYY[j * nx + i];
    tauxyAV[j * nx + i] = (1 - lam[j * nx + i]) * tauxyAV[j * nx + i];
    if (i < nx - 1 && j < ny - 1) {
        tauXY[j * (nx - 1) + i] = (1 - 0.25 * (lam[j * nx + i] + lam[j * nx + i + 1] + lam[(j + 1) * nx + i] + lam[(j + 1) * nx + i + 1])) * tauXY[j * (nx - 1) + i];
    }
    J2[j * nx + i] = sqrt(((tauXX[j * nx + i]) * (tauXX[j * nx + i]) + (tauYY[j * nx + i]) * (tauYY[j * nx + i]))/2 + (tauxyAV[j * nx + i]) * (tauxyAV[j * nx + i]));
    lama[j * nx + i] = lama[j * nx + i] + lam[j * nx + i];
}

__global__ void ComputeSigma(double* Sxx, double* Syy, const double* const P, const double* const tauXX, const double* const tauYY, const long int nx, const long int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Sxx[j * nx + i] = tauXX[j * nx + i] - P[j * nx + i];
    Syy[j * nx + i] = tauYY[j * nx + i] - P[j * nx + i];
}

__global__ void ComputeF(double* Fx, double* Fy, const double* const Sxx, const double* const Syy, const double* const tauXY,
                         const double dx, const double dy, const long int nx, const long int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < (nx - 1) && j < (ny - 2)) {
        Fx[j * (nx - 1) + i] = (Sxx[(j + 1) * nx + i + 1] - Sxx[(j + 1) * nx + i]) / dx + (tauXY[(j + 1) * (nx - 1) + i] - tauXY[j * (nx - 1) + i]) / dy;
    }
    if (i < (nx - 2) && j < (ny - 1)) {
        Fy[j * (nx - 2) + i] = (Syy[(j + 1) * nx + i + 1] - Syy[j * nx + i + 1]) / dy + (tauXY[j * (nx - 1) + i + 1] - tauXY[j * (nx - 1) + i]) / dx;
    }
}

__global__ void ComputeV(double* Vx, double* Vy, double* Ux, const double* const Fx, const double* const Fy, const double rho,
                         const double damp, const double dt, const long int nx, const long int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx - 1  && j < ny - 2) {
        Vx[(j + 1) * (nx + 1) + i + 1] = (Vx[(j + 1) * (nx + 1) + i + 1] + dt / rho * Fx[j * (nx - 1) + i]) / (1 + damp / nx);
    }
    if (i < nx - 2 && j < ny - 1) {
        Vy[(j + 1) * nx + i + 1] = (Vy[(j + 1) * nx + i + 1] + dt / rho * Fy[j * (nx - 2) + i]) / (1 + damp / nx);
    }
    if (i < nx + 1 && j < ny) {
        Ux[j * (nx + 1) + i] = Ux[j * (nx + 1) + i] + Vx[j * (nx + 1) + i] * dt;
    }
}

void SaveMatrix(double* const A_cpu, const double* const A_cuda, const int m, const int n, const std::string filename) {
    cudaMemcpy(A_cpu, A_cuda, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    FILE* A_filw = fopen(filename.c_str(), "wb");
    fwrite(A_cpu, sizeof(double), m * n, A_filw);
    fclose(A_filw);
}

void LoadMatrix(double** A_cpu, double** A_cuda, const int nx, const int ny, const std::string filename) {
    *A_cpu = (double*)malloc(nx * ny * sizeof(double));
    FILE* A_filr = fopen(filename.c_str(), "rb");
    fread(*A_cpu, sizeof(double), (nx) * (ny), A_filr);
    fclose(A_filr);
    cudaMalloc(A_cuda, nx * ny * sizeof(double));
    cudaMemcpy(*A_cuda, *A_cpu, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
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
    /*if there is no matlab*/
    /*int NGRID = 2;*/
    //int NITER = 1000;
    //double K0 = 1;
    //double G0 = 1;
    //double coh = G0*0.01;
    //double rho = 92.6100;
    //double DX = 0.3175;
    //double DY = 0.3175;
    ////double dt = 1;
    //double damp = 5 / 1;


    /*fixing the transfer problem*/
    double K0 = K;
    double G0 = G;
    double coh = COH;
    double rho = RHO;
    double dt = DT;
    double damp = DAMP;
    //double dx = DX;
    /*std::cout << "K0=" << K0 << "     " << K << '\n';
    std::cout << "G0=" << G0 << "     " << G << '\n';
    std::cout << "coh=" << coh << "     " << COH << '\n';
    std::cout << "rho=" << rho << "     " << RHO << '\n';
    std::cout << "dt=" << dt << "     " << DT << '\n';
    std::cout << "damp=" << damp << "     " << DAMP << '\n';
    std::cout << "dx = " << DX << '\n';
    std::cout << "dy = " << DY << '\n';*/
    
    /*FILE* fil = fopen("dx.dat", "wb");
    fwrite( &dx, sizeof(double), 1, fil);
    fclose(fil);*/

    
    dim3 grid, block;
    block.x = 32; grid.x = NGRID;
    block.y = 32; grid.y = NGRID;
    const  long int nx = block.x * grid.x;
    const  long int ny = block.y * grid.y;
    
    cudaSetDevice(0);
    cudaDeviceReset();
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    

    //loading data

    double* Vx_cpu;
    double* Vx_cuda;
    LoadMatrix(&Vx_cpu, &Vx_cuda, nx + 1, ny, "Vx.dat");

    double* Vy_cpu;
    double* Vy_cuda;
    LoadMatrix(&Vy_cpu, &Vy_cuda, nx, ny + 1, "Vy.dat");

    double* tauXX_cpu;
    double* tauXX_cuda;
    LoadMatrix(&tauXX_cpu, &tauXX_cuda, nx, ny, "tauXX.dat");

    double* tauYY_cpu;
    double* tauYY_cuda;
    LoadMatrix(&tauYY_cpu, &tauYY_cuda, nx, ny, "tauYY.dat");

    double* tauXY_cpu;
    double* tauXY_cuda;
    LoadMatrix(&tauXY_cpu, &tauXY_cuda, nx - 1, ny - 1, "tauXY.dat");

    double* P_cpu;
    double* P_cuda;
    LoadMatrix(&P_cpu, &P_cuda, nx, ny, "P.dat");

    double* Ux_cpu;
    double* Ux_cuda;
    LoadMatrix(&Ux_cpu, &Ux_cuda, nx + 1, ny, "Ux.dat");

    //set zeros
    double* tauxyAV_cpu;
    double* tauxyAV_cuda;
    SetMatrixZero(&tauxyAV_cpu, &tauxyAV_cuda, nx, ny);

    double* Exx_cpu;
    double* Exx_cuda;
    SetMatrixZero(&Exx_cpu, &Exx_cuda, nx, ny);

    double* Eyy_cpu;
    double* Eyy_cuda;
    SetMatrixZero(&Eyy_cpu, &Eyy_cuda, nx, ny);

    double* Exy_cpu;
    double* Exy_cuda;
    SetMatrixZero(&Exy_cpu, &Exy_cuda, nx - 1, ny - 1);

    double* J2_cpu;
    double* J2_cuda;
    SetMatrixZero(&J2_cpu, &J2_cuda, nx, ny);

    double* lam_cpu;
    double* lam_cuda;
    SetMatrixZero(&lam_cpu, &lam_cuda, nx, ny);

    double* lama_cpu;
    double* lama_cuda;
    SetMatrixZero(&lama_cpu, &lama_cuda, nx, ny);

    int* flag_cpu;
    int* flag_cuda;
    flag_cpu = (int*)malloc(1*sizeof(int));
    flag_cpu[0] = 0;
    cudaMalloc(&flag_cuda, 1 * sizeof(int));
    cudaMemcpy(flag_cuda, flag_cpu, 1 * sizeof(int), cudaMemcpyHostToDevice);

    double* Sxx_cpu;
    double* Sxx_cuda;
    SetMatrixZero(&Sxx_cpu, &Sxx_cuda, nx, ny);

    double* Syy_cpu;
    double* Syy_cuda;
    SetMatrixZero(&Syy_cpu, &Syy_cuda, nx, ny);

    double* Fx_cpu;
    double* Fx_cuda;
    SetMatrixZero(&Fx_cpu, &Fx_cuda, nx - 1, ny - 2);

    double* Fy_cpu;
    double* Fy_cuda;
    SetMatrixZero(&Fy_cpu, &Fy_cuda, nx - 2, ny - 1);
    //std::cout << "dy = " << DY << '\n';

    /*ACTION LOOP*/
    for (int i = 0; i < NITER; i++) {
        ComputeE<<<grid, block>>>(Exx_cuda, Eyy_cuda, Exy_cuda, Vx_cuda, Vy_cuda, DX, DY, nx, ny);
        cudaDeviceSynchronize();
        //SaveMatrix(P_cpu, P_cuda, nx, ny, "testc.dat");
        //std::cout << "dy = " << DY << '\n';
        ComputeTau<<<grid, block >>>(P_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, Exx_cuda, Eyy_cuda, Exy_cuda, G0, K0, dt, nx, ny);
        cudaDeviceSynchronize();
        /*Plasticity start*/
        //std::cout << "dy = " << DY << '\n';
        flag_cpu[0] = 0;
        cudaMemcpy(flag_cuda, flag_cpu, 1 * sizeof(int), cudaMemcpyHostToDevice);
        ComputeTauxyAv<<<grid, block>>>(tauxyAV_cuda, tauXY_cuda, nx, ny);
        cudaDeviceSynchronize();
        ComputePlast<<<grid, block>>>(J2_cuda, tauXX_cuda, tauYY_cuda, tauXY_cuda, tauxyAV_cuda, lam_cuda, nx, ny, coh, flag_cuda);
        cudaDeviceSynchronize();        
        cudaMemcpy(flag_cpu, flag_cuda, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        if (flag_cpu[0] > 0) {
            ComputePlastCor<<<grid, block>>>(tauXX_cuda, tauYY_cuda, tauXY_cuda, tauxyAV_cuda, J2_cuda, lama_cuda, lam_cuda, nx, ny);
            cudaDeviceSynchronize();
        }
        flag_cpu[0] = 0;        
        cudaMemcpy(flag_cuda, flag_cpu, 1 * sizeof(int), cudaMemcpyHostToDevice);
        /*SaveMatrix(J2_cpu, J2_cuda, nx, ny, "J2c.dat");
        SaveMatrix(lam_cpu, lam_cuda, nx, ny, "lamc.dat");*/
        //std::cout << "dy = " << DY << '\n';
        /*Plasticity end*/
        ComputeSigma<<<grid, block>>>(Sxx_cuda, Syy_cuda, P_cuda, tauXX_cuda, tauYY_cuda, nx, ny);
        cudaDeviceSynchronize();
        //SaveMatrix(Syy_cpu, Syy_cuda, nx, ny, "testc.dat");
        //std::cout << "dy = " << DY << '\n';
        ComputeF<<<grid, block>>>(Fx_cuda, Fy_cuda, Sxx_cuda, Syy_cuda, tauXY_cuda, DX, DY, nx, ny);
        cudaDeviceSynchronize();
        /*SaveMatrix(tauXY_cpu, tauXY_cuda, nx-1, ny-1, "tauXYc.dat");
        SaveMatrix(Sxx_cpu, Sxx_cuda, nx, ny, "Sxxc.dat");
        SaveMatrix(Syy_cpu, Syy_cuda, nx, ny, "Syyc.dat");*/
        /*SaveMatrix(Fx_cpu, Fx_cuda, nx - 1, ny - 2, "testFc.dat");
        SaveMatrix(Fy_cpu, Fy_cuda, nx - 2, ny - 1, "testc.dat");*/
        /*std::cout << "dx = " << DX << '\n';
        std::cout << "dy = " << DY << '\n';*/
        ComputeV<<<grid, block>>>(Vx_cuda, Vy_cuda, Ux_cuda, Fx_cuda, Fy_cuda, rho, damp, dt, nx, ny);
        cudaDeviceSynchronize();
    }

    /*Save Data*/
    SaveMatrix(Vx_cpu, Vx_cuda, nx + 1, ny, "Vxc.dat");
    SaveMatrix(Vy_cpu, Vy_cuda, nx, ny + 1, "Vyc.dat");
    SaveMatrix(Ux_cpu, Ux_cuda, nx + 1, ny, "Uxc.dat");
    SaveMatrix(P_cpu, P_cuda, nx, ny, "Pc.dat");
    SaveMatrix(tauXX_cpu, tauXX_cuda, nx, ny, "tauXXc.dat");
    SaveMatrix(tauYY_cpu, tauYY_cuda, nx, ny, "tauYYc.dat");
    SaveMatrix(tauXY_cpu, tauXY_cuda, nx-1, ny-1, "tauXYc.dat");
    SaveMatrix(tauxyAV_cpu, tauxyAV_cuda, nx, ny, "tauxyAVc.dat");
    SaveMatrix(J2_cpu, J2_cuda, nx, ny, "J2c.dat");
    SaveMatrix(lam_cpu, lam_cuda, nx, ny, "lamc.dat");
    SaveMatrix(lama_cpu, lama_cuda, nx, ny, "lamac.dat");
    SaveMatrix(Sxx_cpu, Sxx_cuda, nx, ny, "Sxxc.dat");
    SaveMatrix(Syy_cpu, Syy_cuda, nx, ny, "Syyc.dat");
    SaveMatrix(Fx_cpu, Fx_cuda, nx - 1, ny - 2, "Fc.dat");
    SaveMatrix(Fy_cpu, Fy_cuda, nx - 2, ny - 1, "Fyc.dat");
    SaveMatrix(Exx_cpu, Exx_cuda, nx, ny, "Exxc.dat");
    SaveMatrix(Eyy_cpu, Eyy_cuda, nx, ny, "Eyyc.dat");
    SaveMatrix(Exy_cpu, Exy_cuda, nx - 1, ny - 1, "Exyc.dat");

    /*Free mem*/
    free(Vx_cpu);
    free(Vy_cpu);
    free(Ux_cpu);
    free(tauXX_cpu);
    free(tauYY_cpu);
    free(tauXY_cpu);
    free(P_cpu);
    free(tauxyAV_cpu);
    free(Exx_cpu);
    free(Eyy_cpu);
    free(Exy_cpu);
    free(J2_cpu);
    free(lam_cpu);
    free(lama_cpu);
    free(flag_cpu);
    free(Sxx_cpu);
    free(Syy_cpu);
    free(Fx_cpu);
    free(Fy_cpu);

    cudaFree(Vx_cuda);
    cudaFree(Vy_cuda);
    cudaFree(Ux_cuda);
    cudaFree(tauXX_cuda);
    cudaFree(tauYY_cuda);
    cudaFree(tauXY_cuda);
    cudaFree(P_cuda);
    cudaFree(tauxyAV_cuda);
    cudaFree(Exx_cuda);
    cudaFree(Eyy_cuda);
    cudaFree(Exy_cuda);
    cudaFree(J2_cuda);
    cudaFree(lam_cuda);
    cudaFree(lama_cuda);
    cudaFree(flag_cuda);
    cudaFree(Sxx_cuda);
    cudaFree(Syy_cuda);
    cudaFree(Fx_cuda);
    cudaFree(Fy_cuda);

    return 0;
}