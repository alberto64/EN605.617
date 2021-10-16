#include <stdio.h>
#include <time.h>

/**
* addCuda: A method that add two arrays and places the result in a third array using 
* multithreading for index calculation.
*/
__global__ void addCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] + randNumList[idx]; 
}

/**
* subCuda: A method that substract two arrays and places the result in a third array using 
* multithreading for index calculation.
*/
__global__ void subCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] - randNumList[idx]; 
}

/**
* multCuda: A method that multiplies two arrays and places the result in a third array using 
* multithreading for index calculation.
*/
__global__ void multCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] * randNumList[idx]; 
}

/**
* modCuda: A method that does the modulus between two arrays and places the result in a third 
* array using multithreading for index calculation.
*/
__global__ void modCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] % randNumList[idx]; 
}

/**
* printArray: A method that takes in an a label and an array with its size and it feeds it to printf.
*/
void printArray(const char* name, int *array, int size) {
	printf("\n%s: [ ", name);
	for(int idx = 0; idx < size; idx++) {
		printf("%i ", array[idx]);
	}
	printf("]\n");
}

/**
* runOperations: Taking the number of blocks and threads it does 4 operations on the two 
* given arrays and prints their results.
*/
void runOperations(int numBlocks, int totalThreads) { 
    
	// Setup Timing Variables
	cudaEvent_t start, stop; 
	float elapsedTimeInMiliseconds; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
  
	// Setup CUDA Streams
	cudaStream_t operationStream; 
	cudaStreamCreate(&operationStream); 
  
	// Setup host and device memory variables
	int *host_threadCountList, *host_randNumberList, *host_addresult, *host_subresult, *host_multresult, *host_modresult; 
	int *device_threadCountList, *device_randNumberList, *device_result;//, *device_subresult, *device_multresult, *device_modresult; 
	cudaMalloc((void**) &device_threadCountList, totalThreads * sizeof(*device_threadCountList)); 
	cudaMalloc((void**) &device_randNumberList, totalThreads * sizeof(*device_randNumberList)); 
	cudaMalloc((void**) &device_result, totalThreads * sizeof(*device_result)); 
	//cudaMalloc((void**) &device_subresult, totalThreads * sizeof(*device_subresult)); 
	//cudaMalloc((void**) &device_multresult, totalThreads * sizeof(*device_multresult)); 
	//cudaMalloc((void**) &device_modresult, totalThreads * sizeof(*device_modresult)); 
  
	cudaHostAlloc((void**) &host_threadCountList, totalThreads * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_randNumberList, totalThreads * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_addresult, totalThreads * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_subresult, totalThreads * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_multresult, totalThreads * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_modresult, totalThreads * sizeof(int), cudaHostAllocDefault);
  
	// Populate host inputs
	for(int idx = 0; idx < totalThreads; idx++) { 
		host_threadCountList[idx] = idx; 
		host_randNumberList[idx] = rand() % 4; 
	} 

	// Turned of to minimize printing
	printArray("Thread Count", host_threadCountList, totalThreads);
	printArray("Random Numbers", host_randNumberList, totalThreads);

	// Start event
	cudaEventRecord(start);

	// Synchronize input variables
	cudaMemcpyAsync(device_threadCountList, host_threadCountList, totalThreads * sizeof(int), cudaMemcpyHostToDevice, operationStream); 
	cudaMemcpyAsync(device_randNumberList, host_randNumberList, totalThreads * sizeof(int), cudaMemcpyHostToDevice, operationStream); 
  
	// Execute each operation and bring result from device to host
	addCUDA<<<totalThreads, numBlocks, 1, operationStream>>>(device_threadCountList, device_randNumberList, device_result);
	cudaMemcpyAsync(host_addresult, device_result, totalThreads * sizeof(int), cudaMemcpyDeviceToHost, operationStream);
	
	subCUDA<<<totalThreads, numBlocks, 1, operationStream>>>(device_threadCountList, device_randNumberList, device_result);
	cudaMemcpyAsync(host_subresult, device_result, totalThreads * sizeof(int), cudaMemcpyDeviceToHost, operationStream);

	multCUDA<<<totalThreads, numBlocks, 1, operationStream>>>(device_threadCountList, device_randNumberList, device_result);
	cudaMemcpyAsync(host_multresult, device_result, totalThreads * sizeof(int), cudaMemcpyDeviceToHost, operationStream);

	modCUDA<<<totalThreads, numBlocks, 1, operationStream>>>(device_threadCountList, device_randNumberList, device_result);
	cudaMemcpyAsync(host_modresult, device_result, totalThreads * sizeof(int), cudaMemcpyDeviceToHost, operationStream);

	cudaStreamSynchronize(operationStream);
  	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop); 
  	cudaEventElapsedTime(&elapsedTimeInMiliseconds, start, stop); 
	printf("Stream and Event Time: %f Miliseconds\n", elapsedTimeInMiliseconds) * 100;

	// Turned of to minimize printing
	printArray("Add Result", host_addresult, totalThreads);
	printArray("Sub Result", host_subresult, totalThreads);
	printArray("Mult Result", host_multresult, totalThreads);
	printArray("Mod Result", host_modresult, totalThreads);

	// Free reserved memory
	cudaFreeHost(host_threadCountList);
	cudaFreeHost(host_randNumberList);
	cudaFreeHost(host_addresult);
	cudaFreeHost(host_subresult);
	cudaFreeHost(host_multresult);
	cudaFreeHost(host_modresult);
	cudaFree(device_threadCountList);
	cudaFree(device_randNumberList);
	cudaFree(device_result);
}

int main(int argc, char** argv) {
	// Based on the work of Andrew Krepps
	
	// Set default values in case arguments don't come in command line.
	int totalThreads = 1024;
	int blockSize = 256;

	// read command line arguments
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	printf("\nTotal Threads: %d\nBlock Size: %d\n", totalThreads, blockSize);
	
	runOperations(numBlocks, totalThreads);

	return 0;
}
