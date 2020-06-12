// Wave 2D 
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include "cuda.h"

#define end                  }  
#define load(A,nx,ny,Aname)  double *A##_d,*A##_h; A##_h = (double*)malloc((nx)*(ny)*sizeof(double));  \
                             FILE* A##fid=fopen(Aname, "rb"); fread(A##_h, sizeof(double), (nx)*(ny), A##fid); fclose(A##fid); \
                             cudaMalloc(&A##_d,((nx)*(ny))*sizeof(double)); \
                             cudaMemcpy(A##_d,A##_h,((nx)*(ny))*sizeof(double),cudaMemcpyHostToDevice);                             
#define save(A,nx,ny,Aname)  cudaMemcpy(A##_h,A##_d,((nx)*(ny))*sizeof(double),cudaMemcpyDeviceToHost);  \
                             FILE* A##fidw=fopen(Aname, "wb"); fwrite(A##_h, sizeof(double), ((nx)*(ny)), A##fidw); fclose(A##fidw);                          
#define for_(ix_start,ix_end,iy_start,iy_end) if (ix>=(ix_start) && ix<=(ix_end)  && iy>=(iy_start) && iy<=(iy_end)){
               
#define fun_call(A)          A<<<grid,block>>>(      So_d,      Vx_d,        Vy_d,        Pr_d,        pa_d,            nx,            ny); cudaDeviceSynchronize();   
#define fun_def(A)           __global__ void A(double* So,double* Vx,double* Vy  ,double* Pr  , double* pa , const  int nx, const  int ny){ \
                             int ix = blockIdx.x*blockDim.x + threadIdx.x + 1; /*if (ix>=(nx+1)) return;*/ \
                             int iy = blockIdx.y*blockDim.y + threadIdx.y + 1; /*if (iy>=(ny+1)) return;*/ 
#define  Pres(ix,iy)          Pr[ix + (iy-1)*nx     - 1]  
#define  Velx(ix,iy)          Vx[ix + (iy-1)*(nx+1) - 1]         
#define  Vely(ix,iy)          Vy[ix + (iy-1)*nx     - 1]  
#define  Sorc(ix,iy)          So[ix + (iy-1)*nx     - 1]    
#define NGRID  1
#define NPARS  6
#define NT  10
//#define  para(ix,iy)          pa[ix                 - 1] 
                             
fun_def(compute_V) 
double dx=pa[0],dy=pa[1],dt=pa[2],rho=pa[4],dmpX=pa[5],dmpY=pa[6];



//double dx=para(1),dy=pa[1],dt=pa[2],rho=pa[4],dmpX=pa[5],dmpY=pa[6];
    for_(2,nx  ,2,ny-1) Velx(ix,iy) = Velx(ix,iy)*dmpX - dt*(Pres(ix,iy)-Pres(ix-1,iy  ))/dx/rho; end 
    for_(2,nx-1,2,ny  ) Vely(ix,iy) = Vely(ix,iy)*dmpY - dt*(Pres(ix,iy)-Pres(ix  ,iy-1))/dy/rho; end
    printf("dt = %lf\n", dt);
end


            
fun_def(compute_P)
double dx=pa[0],dy=pa[1],dt=pa[2],k=pa[3];//,dmpX=pa[5];
          Pres(ix,iy) = Pres(ix,iy)       - dt*k*((Velx(ix+1,iy  )-Velx(ix,iy))/dx
                                          +       (Vely(ix  ,iy+1)-Vely(ix,iy))/dy );//  + dt*dt * Sorc(ix,iy);
          /*if (ix ==  1) Pres(ix,iy) =  0*iy; 
          if (iy ==  1) Pres(ix,iy) =  0*ix; 
          if (ix == nx) Pres(ix,iy) =  0*nx; 
          if (iy == ny) Pres(ix,iy) =  0*ny; */
end

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

  /*double* pa_cpu2 = (double*)malloc(NPARS * sizeof(double));
  cudaMemcpy(pa_cpu2, pa_cuda, NPARS * sizeof(double), cudaMemcpyDeviceToHost);

  std::cout << "dT = " << pa_cpu2[2] << std::endl;*/

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

  /*load(pa,NPARS,  1,"pa.dat")
  load(Pr,nx  ,ny  ,"Pr.dat")
  load(Vx,nx+1,ny  ,"Vx.dat")
  load(Vy,nx  ,ny+1,"Vy.dat")
  load(So,nx  ,ny  ,"So.dat")
  double dt = pa_h[2];
  printf("dt = %lf\n", dt);
  for (int it=1;it<=NT;it++){  
  fun_call(compute_V)
  fun_call(compute_P)
  end        
  save(Pr,nx  ,ny  ,"Pr.dat")
  save(Vx,nx+1,ny  ,"Vx.dat")
  save(Vy,nx  ,ny+1,"Vy.dat")*/
  cudaDeviceReset();
  return 0;
}
