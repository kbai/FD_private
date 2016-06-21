#include "utils.hh"
#include "finitedifference.cuh"


class fd_solver
{
	public:
		fd_solver(int Nx, int Nz, realw deltat, int myrank, cudaStream_t custream);

		~fd_solver();

		void update_stress_veloc();

		void write_snapshot(int t, int rank);

		realw* get_left_column_sxx();

		realw* get_right_column_sxx();

		realw* get_left_column_sxz();

		realw* get_right_column_sxz();

		realw* get_left_column_szz();

		realw* get_right_column_szz();

		realw* get_left_column_u();

		realw* get_right_column_u();

		realw* get_left_column_w();

		realw* get_right_column_w();

		void communicate(fd_solver *left, fd_solver *right, bool has_left, bool has_right);







	private:
		int Lx, Lz;
		realw dt;			/** timestep */
		int myrank;
		cudaStream_t my_stream;
	
		
		realw *veloc_u;		/** horizontal velocity */
		realw *dev_veloc_u;
		realw *veloc_w;		/** vertical velocity */
		realw *dev_veloc_w;
		realw *dev_stress_xx;		/** stress components */
		realw *dev_stress_zz;
		realw *dev_stress_xz;
		realw *dev_ux;
		realw *dev_wx;
		realw *dev_uz;
		realw *dev_wz;
		realw *dev_xx_x, *dev_xz_z, *dev_xz_x, *dev_zz_z;

		realw *dev_lambda, *dev_mu, *dev_rho, *lambda, *mu, *rho;		/** material properties*/
		dim3 num_blocks_x;
		dim3 thread_per_block; 

		realw* getleft(realw* field)
		{
			return field;
		}
		realw* getright(realw* field)
		{
			return field + (Lx-1) * (Lz);
		}


};
fd_solver::fd_solver(int Nx, int Nz, realw deltat, int myrank, cudaStream_t custream)
{
	this->Lx = Nx;
	this->Lz = Nz;
	dt = deltat;
	this->myrank = myrank;
	this->my_stream = custream;
	num_blocks_x = dim3(16,16,1);
	thread_per_block = dim3(16,32,1);

	int field_size = Nx*Nz*sizeof(realw);
	veloc_u = (realw*)malloc(field_size);
	veloc_w = (realw*)malloc(field_size);
	lambda	= (realw*)malloc(field_size);
	mu	= (realw*)malloc(field_size);
	rho = (realw*)malloc(field_size);
	utils::set_constant_value(lambda, 1.0, Nx*Nz);
	utils::set_constant_value(mu, 1.0, Nx*Nz);
	utils::set_constant_value(rho, 1.0, Nx*Nz);

	if(myrank == 5)
	{
		utils::set_gaussian_value(veloc_u, 0.0, Nx, Nz);
	}
	else
	{
		utils::set_constant_value(veloc_u, 0.0, Nx*Nz);
	}
	utils::set_constant_value(veloc_w, 0.0, Nx*Nz);
	
	gpuErrchk(cudaMalloc((void**)&dev_veloc_u,field_size));
	gpuErrchk(cudaMalloc((void**)&dev_veloc_w,field_size));

	gpuErrchk(cudaMalloc((void**)&dev_lambda, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_mu, field_size));
   	gpuErrchk(cudaMalloc((void**)&dev_rho,field_size));
	gpuErrchk(cudaMalloc((void**)&dev_stress_xx, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_stress_zz, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_stress_xz, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_xx_x, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_xz_z, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_xz_x, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_zz_z, field_size));

	gpuErrchk(cudaMemset(dev_stress_xx, 0, field_size));
	gpuErrchk(cudaMemset(dev_stress_zz, 0, field_size));
	gpuErrchk(cudaMemset(dev_stress_xz, 0, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_ux, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_uz, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_wx, field_size));
	gpuErrchk(cudaMalloc((void**)&dev_wz, field_size));
	gpuErrchk(cudaMemset(dev_xx_x, 0, field_size));
	gpuErrchk(cudaMemset(dev_xz_z, 0, field_size));
	gpuErrchk(cudaMemset(dev_xz_x, 0, field_size));
	gpuErrchk(cudaMemset(dev_zz_z, 0, field_size));
	gpuErrchk(cudaMemset(dev_ux, 0, field_size));
	gpuErrchk(cudaMemset(dev_uz, 0, field_size));
	gpuErrchk(cudaMemset(dev_wx, 0, field_size));
	gpuErrchk(cudaMemset(dev_wz, 0, field_size));

	gpuErrchk(cudaMemcpy(dev_veloc_u, veloc_u, field_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_veloc_w, veloc_w, field_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_lambda, lambda, field_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_mu, mu, field_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_rho, rho, field_size, cudaMemcpyHostToDevice));

}

fd_solver::~fd_solver()
{
	cudaFree(dev_veloc_u);
	cudaFree(dev_veloc_w);
	cudaFree(dev_stress_xx);
	cudaFree(dev_stress_xz);
	cudaFree(dev_stress_zz);
	cudaFree(dev_mu);
	cudaFree(dev_lambda);
	cudaFree(dev_rho);
	cudaFree(dev_ux);
	cudaFree(dev_uz);
	cudaFree(dev_wx);
	cudaFree(dev_wz);
	cudaFree(dev_xx_x);
	cudaFree(dev_zz_z);
	cudaFree(dev_xz_x);
	cudaFree(dev_xz_z);
	free(veloc_u);
	free(veloc_w);
	free(lambda);
	free(mu);
	free(rho);

}

void fd_solver::update_stress_veloc()
{
	compute_derivative(XDERIVATIVE,  dev_stress_xx,  dev_xx_x, Lx, Lz, num_blocks_x, thread_per_block, my_stream);
	compute_derivative(XDERIVATIVE,  dev_stress_xz,  dev_xz_x, Lx, Lz, num_blocks_x, thread_per_block, my_stream);
	compute_derivative(ZDERIVATIVE,  dev_stress_zz,  dev_zz_z, Lx, Lz, num_blocks_x, thread_per_block, my_stream);
	compute_derivative(ZDERIVATIVE,  dev_stress_xz,  dev_xz_z, Lx, Lz, num_blocks_x, thread_per_block, my_stream);
	compute_derivative(XDERIVATIVE,  dev_veloc_u,  dev_ux, Lx, Lz, num_blocks_x, thread_per_block, my_stream);
	compute_derivative(XDERIVATIVE,  dev_veloc_w,  dev_wx, Lx, Lz, num_blocks_x, thread_per_block, my_stream);
	compute_derivative(ZDERIVATIVE,  dev_veloc_u,  dev_uz, Lx, Lz, num_blocks_x, thread_per_block, my_stream);
	compute_derivative(ZDERIVATIVE,  dev_veloc_w,  dev_wz, Lx, Lz, num_blocks_x, thread_per_block, my_stream);
	call_update_field(dev_veloc_u, dev_veloc_w, dev_stress_xx, dev_stress_zz, dev_stress_xz,
	dev_xx_x, dev_zz_z, dev_xz_x, dev_xz_z, dev_ux, dev_uz, dev_wx, dev_wz, dev_lambda, dev_mu,
	dev_rho, dt, Lx, Lz, num_blocks_x, thread_per_block, my_stream);

}

void fd_solver::write_snapshot(int t, int rank)
{
	utils::output_field(dev_veloc_u, veloc_u, t, rank,  Lx, Lz);
}

realw* fd_solver::get_left_column_sxx()
{
	return this->getleft(this->dev_stress_xx);
}

realw* fd_solver::get_right_column_sxx()
{
	return this->getright(this->dev_stress_xx);
}

realw* fd_solver::get_left_column_szz()
{
	return this->getleft(this->dev_stress_zz);
}

realw* fd_solver::get_right_column_szz()
{
	return this->getright(this->dev_stress_zz);
}


realw* fd_solver::get_left_column_sxz()
{
	return this->getleft(this->dev_stress_xz);
}

realw* fd_solver::get_right_column_sxz()
{
	return this->getright(this->dev_stress_xz);
}


realw* fd_solver::get_left_column_u()
{
	return this->getleft(this->dev_veloc_u);
}

realw* fd_solver::get_right_column_u()
{
	return this->getright(this->dev_veloc_u);
}

realw* fd_solver::get_left_column_w()
{
	return this->getleft(this->dev_veloc_w);
}

realw* fd_solver::get_right_column_w()
{
	return this->getright(this->dev_veloc_w);
}

void fd_solver::communicate(fd_solver *left, fd_solver *right, bool has_left, bool has_right)
{
	realw* tmp;

	if(has_left)
	{
		tmp = left->get_right_column_sxx();
		gpuErrchk(cudaMemcpy(tmp, dev_stress_xx + Lz, Lz* sizeof(realw),cudaMemcpyDefault));
		tmp = left->get_right_column_sxz();
		gpuErrchk(cudaMemcpy(tmp, dev_stress_xz + Lz, Lz* sizeof(realw),cudaMemcpyDefault));
		tmp = left->get_right_column_szz();
		gpuErrchk(cudaMemcpy(tmp, dev_stress_zz + Lz, Lz* sizeof(realw),cudaMemcpyDefault));
		tmp = left->get_right_column_u();
		gpuErrchk(cudaMemcpy(tmp, dev_veloc_u + Lz, Lz* sizeof(realw),cudaMemcpyDefault));
		tmp = left->get_right_column_w();
		gpuErrchk(cudaMemcpy(tmp, dev_veloc_w + Lz, Lz* sizeof(realw),cudaMemcpyDefault));
	}
	if(has_right)
	{
		tmp = right->get_left_column_sxx();
		gpuErrchk(cudaMemcpy(tmp, dev_stress_xx + (Lx-2)*Lz, Lz* sizeof(realw),cudaMemcpyDefault));
		tmp = right->get_left_column_sxz();
		gpuErrchk(cudaMemcpy(tmp, dev_stress_xz + (Lx-2)*Lz, Lz* sizeof(realw),cudaMemcpyDefault));
		tmp = right->get_left_column_szz();
		gpuErrchk(cudaMemcpy(tmp, dev_stress_zz + (Lx-2)*Lz, Lz* sizeof(realw),cudaMemcpyDefault));
		tmp = right->get_left_column_u();
		gpuErrchk(cudaMemcpy(tmp, dev_veloc_u + (Lx-2)*Lz, Lz* sizeof(realw),cudaMemcpyDefault));
		tmp = right->get_left_column_w();
		gpuErrchk(cudaMemcpy(tmp, dev_veloc_w + (Lx-2)*Lz, Lz* sizeof(realw),cudaMemcpyDefault));
	}
	

}






