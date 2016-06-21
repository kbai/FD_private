#include <math.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#define gpuErrchk(ans) { utils::gpuAssert((ans), __FILE__, __LINE__); }
typedef float realw;

namespace utils
{
//void output_field(reaw*, realw* , int , int , int , int);
//void set_constant_value(realw* , realw, int);
//void set_gaussian_value(realw* , realw , int, int);

inline void gpuAssert(cudaError_t code, const char *file, int line,
		                      bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,line);
		exit(code);
		}
		 
}




void output_field(realw *dev_field, realw *host_field, int t, int rank, int Xm, int Zm)
{
	int field_size = Xm*Zm*sizeof(realw);
	std::cout<<"size"<<field_size<<std::endl;
	gpuErrchk(cudaMemcpy(host_field, dev_field, field_size, cudaMemcpyDeviceToHost));
	std::ofstream myfile;
	std::string filename="proc"+std::to_string(rank)+"snapshot"+std::to_string(t)+".dat";
    
    myfile.open(filename);
    for(int i = 0; i<Xm; i++)
    {
        for(int j=0; j<Zm; j++)
        {
            myfile<<host_field[i*Zm+j]<<'\t';
        }
        myfile << std::endl;
    }
    myfile.close();
    
}

void set_constant_value(float *field, float value, int size)
{
	for(int i = 0 ; i < size; i++)
	{
		field[i] = value;
	}
}

void set_gaussian_value(float *field, float value, int Xm, int Zm)
{
	int i,j;
	for(i = 0 ; i < Xm ; i++)
	{
		for(j=0 ; j < Zm; j++)
		{
			field[i*Zm+j] = exp(-0.01*((i-Xm*0.5)*(i-Xm*0.5)+(j - Zm*0.5)*(j-Zm*0.5)));
		}
	}
}

}
