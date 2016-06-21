#define XDERIVATIVE 1
#define ZDERIVATIVE 2

void compute_derivative(int direction, float *inputfield, float *derivative, int Xm, int Zm, dim3 num_blocks, dim3 thread_per_block, cudaStream_t custream);



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
                       float* uz_z,
                       float* lambda,
                       float* mu,
                       float* rho,
                       float dt,
                       int Xm,
                       int Zm,
                       dim3 num_blocks,
                       dim3 thread_per_block,
					   cudaStream_t custream);



