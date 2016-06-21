//
//  main.cpp
//  FD
//
//  Created by KangchenBai on 5/21/16.
//  Copyright (c) 2016 KangchenBai. All rights reserved.
//
#include "fd.hh"
#include "ta_utilities.hpp"
#include <time.h>




int main(int argc, const char * argv[]) {
    
    // insert code here...
    
    
    float dt = 0.01;
    int Xm = 100;
    int Zm = 500;
    
    int Nstep = 10;
	float start,end;
	int numgpus;
	cudaStream_t custream[10];
    numgpus = TA_Utilities::select_all_GPUs();
//	numgpus = 3;
    std::cout << numgpus<<std::endl;
	fd_solver *fd_obj[10];
	


	for(int i = 0 ; i < 10; i++)
	{
		cudaSetDevice(i%numgpus);
		cudaStreamCreate(&custream[i]);
		fd_obj[i] = new fd_solver(Xm,Zm,dt,i,custream[i]);

	}
    

	start = clock();
    
    for(int t = 0 ; t < Nstep; t++)
    {
	    	std::cout<<t<<std::endl;
    	for(int i = 0 ; i < 10; i++)
		{
			cudaSetDevice(i%numgpus);
			fd_obj[i]->update_stress_veloc();
		}
        for(int i = 1 ; i < 9; i++)
		{
			cudaSetDevice(i%numgpus);
			fd_obj[i]->communicate(fd_obj[i-1],fd_obj[i+1],true,true);
		}
		cudaSetDevice(0);
		fd_obj[0]->communicate(NULL,fd_obj[1],false,true);
		fd_obj[9]->communicate(fd_obj[8],NULL,true,false);
	if(t%1000 == 0)
	{
	for(int i = 0 ; i < 10; i++)
		{
			fd_obj[i]->write_snapshot(t,i);
		}
	}

    }
	std::cout <<"time elapsed:"<< clock()-start <<std::endl;
        
//	fd_obj.~fd_solver();


    
    
    
    
    
    return 0;
}


