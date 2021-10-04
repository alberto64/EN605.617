#include <stdio.h>
#include <time.h>

/**
* addSharedCuda: A method that add two arrays and places the result in a third array using 
* multithreading for index calculation using shared memory.
*/
__device__ void addSharedCUDA(int idx, int *threadCountList, int *randNumList, int *resultList) { 
	resultList[idx] = threadCountList[idx] + randNumList[idx]; 
}

/**
* subSharedCuda: A method that substract two arrays and places the result in a third array using 
* multithreading for index calculation using shared memory.
*/
__device__ void subSharedCUDA(int idx, int *threadCountList, int *randNumList, int *resultList) { 
	resultList[idx] = threadCountList[idx] - randNumList[idx]; 
}

/**
* multSharedCuda: A method that multiplies two arrays and places the result in a third array using 
* multithreading for index calculation using shared memory.
*/
__device__ void multSharedCUDA(int idx, int *threadCountList, int *randNumList, int *resultList) { 
	resultList[idx] = threadCountList[idx] * randNumList[idx]; 
}

/**
* modSharedCuda: A method that does the modulus between two arrays and places the result in a third 
* array using multithreading for index calculation using shared memory.
*/
__device__ void modSharedCUDA(int idx, int *threadCountList, int *randNumList, int *resultList) { 
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
*
*/
__global__ void runOperations(int *dev_threadCountList, int *dev_randNumList, int *dev_addResultList, int *dev_subResultList, int *dev_multResultList, int *dev_modResultList) {
	
	__shared__ int sharedThreadCountList[1000];
	__shared__ int sharedRandNumList[1000];

	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 

	sharedThreadCountList[idx] = dev_threadCountList[idx];
	sharedRandNumList[idx] = dev_randNumList[idx];

	__syncthreads();

	addSharedCUDA(idx, sharedThreadCountList, sharedRandNumList, dev_addResultList);
	subSharedCUDA(idx, sharedThreadCountList, sharedRandNumList, dev_subResultList);
	multSharedCUDA(idx, sharedThreadCountList, sharedRandNumList, dev_multResultList);
	modSharedCUDA(idx, sharedThreadCountList, sharedRandNumList, dev_modResultList);
}

/**
* runOperations: Taking the number of blocks and threads it does 4 operations on the two 
* given arrays and prints their results. Uses shared memory.
*/
void prepareRun(int numBlocks, int totalThreads, int* threadCountList, int* randNumList) { 

	// Prepare result array variables
	int* addresultList = (int*) malloc(totalThreads * sizeof(int));
	int* subresultList = (int*) malloc(totalThreads * sizeof(int));
	int* multresultList = (int*) malloc(totalThreads * sizeof(int));
	int* modresultList = (int*) malloc(totalThreads * sizeof(int));
	
	// Prepare cuda variables
	int *dev_threadCountList, *dev_randNumList, *dev_addResultList, *dev_subResultList, *dev_multResultList, *dev_modResultList;
	cudaMalloc((void**)&dev_threadCountList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_randNumList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_addResultList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_subResultList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_multResultList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_modResultList, totalThreads * sizeof(int));

	// Copy inputs into device memory 
	cudaMemcpy(dev_threadCountList, threadCountList, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randNumList, randNumList, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	
	runOperations<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_addResultList, dev_subResultList, dev_multResultList, dev_modResultList);

	// Bring result from device to host
	cudaMemcpy(addresultList, dev_addResultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 
	cudaMemcpy(subresultList, dev_subResultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 
	cudaMemcpy(multresultList, dev_multResultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 
	cudaMemcpy(modresultList, dev_modResultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	// Turned of to minimize printing
	// printArray("Add Result", addresultList, totalThreads);
	// printArray("Sub Result", subresultList, totalThreads);
	// printArray("Mult Result", multresultList, totalThreads);
	// printArray("Mod Result", modresultList, totalThreads);
	
	// Free reserved memory
	cudaFree(dev_threadCountList);
	cudaFree(dev_randNumList);
	cudaFree(dev_addResultList);
	cudaFree(dev_subResultList);
	cudaFree(dev_multResultList);
	cudaFree(dev_modResultList);
}

/**
* runTest: Used to set up variables needed to run a timing test.
*/ 
void timeTest(const int numBlocks, const int totalThreads) {

	// Set up variables for timing
	clock_t start, end;
	double timePassedMiliSeconds;

	// Set up paged memory space 
	int* threadCountList = (int*) malloc(totalThreads * sizeof(int));
	int* randNumList = (int*) malloc(totalThreads * sizeof(int));

	// Populate paged memory arrays
	for ( int idx = 0; idx < totalThreads; idx++ ) {
    	threadCountList[idx] = idx; 
		randNumList[idx] = rand() % 4;
   	}
	
	// Show generated values
	// Turned oof to minimize printing
	// printArray("Thread Count List", threadCountList, totalThreads);
	// printArray("Random Number List", randNumList, totalThreads);
	
	// Run and time operations using const memory
	start = clock();
	prepareRun(numBlocks, totalThreads, threadCountList, randNumList);
	end = clock();
	timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("Shared Memory Time: %f Miliseconds\n", timePassedMiliSeconds);
}

/**
* Main method: starts the execution.
*/
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

	timeTest(numBlocks, totalThreads);

	return 0;
}