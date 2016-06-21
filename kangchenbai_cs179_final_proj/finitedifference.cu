#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include"finitedifference.cuh"

#define field(a,b)      field[ (a)*Zm + (b) ]
#define diffield(a,b)   diffield[ (a)*Zm + (b) ]
#define coef(a,b)       coef[ (a)*Zm + (b) ]
#define difffield(a,b)  difffield[ (a)*Zm + (b)]


__global__ void compute_spatial_X_derivative(float* difffield, float* field, int Xm, int Zm)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    while(tx < Xm-2)
    {
        
        
        while(ty < Zm-2)
        {
            
            difffield(tx+1,ty+1) = 0.5*( field( tx + 2, ty + 1 )-field( tx ,ty + 1 ));
	    //printf("%d: %d => %f, %f, %f, %d, %d\n",tx,ty,difffield(tx+1,ty+1),field( tx + 2, ty + 1 ), field( tx ,ty + 1 ),Xm,Ym);
	    //printf("%d: %d=> %f\n",tx,ty,field(tx,ty));
            
            ty += blockDim.y * gridDim.y;
        }
        
        
        tx += blockDim.x * gridDim.x;
    }
    
}







__global__ void compute_spatial_Z_derivative(float* difffield, float* field, int Xm, int Zm)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    while(tx < Xm - 2)
    {
        
        
        while(ty < Zm - 2)
        {
            
            difffield( tx + 1,ty + 1 ) = 0.5*(field( tx + 1 , ty + 2 )-field( tx + 1 ,ty ));
            
            ty += blockDim.y * gridDim.y;
        }
        
        
        tx += blockDim.x*gridDim.x;
    }
    
}


__global__ void update_field(float* dev_veloc_u,
                             float* dev_veloc_w,
                             float *stress_xx,
                             float *stress_zz,
                             float *stress_xz,
                             float *dev_xx_x,
                             float *dev_zz_z,
                             float *dev_xz_x,
                             float *dev_xz_z,
                             float *dev_ux_x,
                             float *dev_ux_z,
                             float *dev_uz_x,
                             float *dev_uz_z,
                             float *lambda,
                             float *mu,
                             float *rho,
                             float dt,
                             int Xm,
                             int Zm)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int thread = tx*Zm + ty;
    while(tx < Xm )
    {
        
        
        while(ty < Zm )
        {
            stress_xx[thread] += dt*(dev_ux_x[thread]*(lambda[thread]+2.0*mu[thread]) + dev_uz_z[thread]*lambda[thread]);
            stress_zz[thread] += dt*(dev_uz_z[thread]*(lambda[thread]+2.0*mu[thread]) + dev_ux_x[thread]*lambda[thread]);
            stress_xz[thread] += dt*mu[thread]*(dev_uz_x[thread] + dev_ux_z[thread]);
            dev_veloc_u[thread] += dt*(1./rho[thread])*(dev_xx_x[thread] + dev_xz_z[thread]);
            dev_veloc_w[thread] += dt*(1./rho[thread])*(dev_zz_z[thread] + dev_xz_x[thread]);
            ty += blockDim.y * gridDim.y;
        }
        tx += blockDim.x*gridDim.x;
    }
    
}

void compute_derivative(int direction, float *inputfield, float *derivative, int Xm, int Zm, dim3 num_blocks, dim3 thread_per_block, cudaStream_t custream)
{
    if(direction == XDERIVATIVE)
    {
	//printf("compute X derivative!\n");
        compute_spatial_X_derivative<<<num_blocks, thread_per_block, 0, custream>>>(derivative, inputfield, Xm, Zm);
	cudaError_t error = cudaGetLastError();
	  if(error != cudaSuccess)
		    {
			        // print the CUDA error message and exit
				     printf("CUDA error: %s\n", cudaGetErrorString(error));
				         exit(-1);
	           }
    }
    if(direction == ZDERIVATIVE)
    {
        compute_spatial_Z_derivative<<<num_blocks, thread_per_block,0, custream>>>(derivative, inputfield, Xm, Zm);
	cudaError_t error = cudaGetLastError();
	  if(error != cudaSuccess)
		    {
			        // print the CUDA error message and exit
				     printf("CUDA error: %s\n", cudaGetErrorString(error));
				         exit(-1);
	           }

    }
}

void call_update_field(float* dev_veloc_u,
                       float* dev_veloc_w,
                       float *stress_xx,
                       float *stress_zz,
                       float *stress_xz,
                       float* dev_xx_x,
                       float* dev_zz_z,
                       float* dev_xz_x,
                       float* dev_xz_z,
                       float* dev_ux_x,
                       float* dev_ux_z,
                       float* dev_uz_x,
		       float* dev_uz_z,
                       float* lambda,
                       float* mu,
                       float* rho,
                       float dt,
                       int Xm,
                       int Zm,
                       dim3 num_blocks,
                       dim3 thread_per_block,
					   cudaStream_t custream)
{
    update_field<<<num_blocks, thread_per_block, 0, custream>>>( dev_veloc_u,
                                                   dev_veloc_w,
                                                   stress_xx,
                                                   stress_zz,
                                                   stress_xz,
                                                   dev_xx_x,
                                                   dev_zz_z,
                                                   dev_xz_x,
                                                   dev_xz_z,
                                                   dev_ux_x,
                                                   dev_ux_z,
                                                   dev_uz_x,
                                                   dev_uz_z,
                                                   lambda,
                                                   mu,
                                                   rho,
                                                   dt,
                                                   Xm,
                                                   Zm);
    
	cudaError_t error = cudaGetLastError();
	  if(error != cudaSuccess)
		    {
			        // print the CUDA error message and exit
				     printf("CUDA error: %s\n", cudaGetErrorString(error));
				         exit(-1);
	           }

}


