#include <stdio.h>
#include <time.h>

/**
* addCuda: A method that add two arrays and places the result in a third array using 
* multithreading for index calculation.
*/
__global__ void addCUDA(const int *threadCountList, const int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] + randNumList[idx]; 
}

/**
* subCuda: A method that substract two arrays and places the result in a third array using 
* multithreading for index calculation.
*/
__global__ void subCUDA(const int *threadCountList, const int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] - randNumList[idx]; 
}

/**
* multCuda: A method that multiplies two arrays and places the result in a third array using 
* multithreading for index calculation.
*/
__global__ void multCUDA(const int *threadCountList, const int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] * randNumList[idx]; 
}

/**
* modCuda: A method that does the modulus between two arrays and places the result in a third 
* array using multithreading for index calculation.
*/
__global__ void modCUDA(const int *threadCountList, const int *randNumList, int *resultList) { 
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
* given arrays and prints their results. Uses paged memory
*/
void runOperations(int numBlocks, int totalThreads, int* threadCountList, int* randNumList) { 
	
	// Prepare result array variables
	int* addresultList = (int*) malloc(totalThreads * sizeof(int));
	int* subresultList = (int*) malloc(totalThreads * sizeof(int));
	int* multresultList = (int*) malloc(totalThreads * sizeof(int));
	int* modresultList = (int*) malloc(totalThreads * sizeof(int));
	
	// Prepare cuda variables
	int* dev_threadCountList, *dev_randNumList, *dev_resultList;
	cudaMalloc((void**)&dev_threadCountList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_randNumList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_resultList, totalThreads * sizeof(int));

	// Copy inputs into device memory 
	cudaMemcpy(dev_threadCountList, threadCountList, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randNumList, randNumList, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	
	// Execute each operation and bring result from device to host
	addCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	cudaMemcpy(addresultList, dev_resultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	subCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	cudaMemcpy(subresultList, dev_resultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	multCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	cudaMemcpy(multresultList, dev_resultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	modCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	cudaMemcpy(modresultList, dev_resultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	// Turned of to minimize printing
	// printArray("Add Result", addresultList, totalThreads);
	// printArray("Sub Result", subresultList, totalThreads);
	// printArray("Mult Result", multresultList, totalThreads);
	// printArray("Mod Result", modresultList, totalThreads);
	
	// Free reserved memory
	cudaFree(dev_threadCountList);
	cudaFree(dev_randNumList);
	cudaFree(dev_resultList);
}

void timeTest(int numBlocks, int totalThreads) {
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
	// Turned of to minimize printing
	// printArray("Thread Count List", threadCountList, totalThreads);
	// printArray("Random Number List", randNumList, totalThreads);
	
	// Run and time operations using paged memory
	start = clock();
	runOperations(numBlocks, totalThreads, threadCountList, randNumList);
	end = clock();
	timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("\nPaged Memory Time: %f Miliseconds\n", timePassedMiliSeconds);
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

	printf("Total Threads: %d\nBlock Size: %d\n", totalThreads, blockSize);
	
	timeTest(numBlocks, totalThreads);

	return 0;
}
