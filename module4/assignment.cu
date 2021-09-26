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

// __global__ void ceasarCypherEncryptCUDA(int cypherKey, char *stringToEncrypt, char *resultString) { 
// 	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
// 	resultString[idx] = stringToEncrypt[idx] + cypherKey; 
// }

// __global__ void ceasarCypherDecryptCUDA(int cypherKey, char *stringToDecrypt, char *resultString) { 
// 	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
// 	resultString[idx] = stringToDecrypt[idx] - cypherKey; 
// }

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
void runOperations(int numBlocks, int totalThreads, int* threadCountList, int* randNumList) { 
	
	int* addresultList = (int*) malloc(totalThreads * sizeof(int));
	int* subresultList = (int*) malloc(totalThreads * sizeof(int));
	int* multresultList = (int*) malloc(totalThreads * sizeof(int));
	int* modresultList = (int*) malloc(totalThreads * sizeof(int));
	int* dev_threadCountList, *dev_randNumList, *dev_resultList;

	cudaMalloc((void**)&dev_threadCountList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_randNumList, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_resultList, totalThreads * sizeof(int));

	cudaMemcpy(dev_threadCountList, threadCountList, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randNumList, randNumList, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	
	addCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	cudaMemcpy(addresultList, dev_resultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	subCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	cudaMemcpy(subresultList, dev_resultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	multCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	cudaMemcpy(multresultList, dev_resultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	modCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	cudaMemcpy(modresultList, dev_resultList, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	//printArray("Add Result", addresultList, totalThreads);
	//printArray("Sub Result", subresultList, totalThreads);
	//printArray("Mult Result", multresultList, totalThreads);
	//printArray("Mod Result", modresultList, totalThreads);
	
	cudaFree(dev_threadCountList);

	cudaFree(dev_randNumList);
	
	cudaFree(dev_resultList);
}

int main(int argc, char** argv)
{
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

	// Set up variables for timing
	clock_t start, end;
	double timePassedMiliSeconds;

	// Set up paged memory space 
	int* threadCountList = (int*) malloc(totalThreads * sizeof(int));
	int* randNumList = (int*) malloc(totalThreads * sizeof(int));
	
	// Set up pinned memory space
	int* pinned_threadCountList;
	int* pinned_randNumList;
	cudaMallocHost((void**)&pinned_threadCountList, totalThreads * sizeof(int));
	cudaMallocHost((void**)&pinned_randNumList, totalThreads * sizeof(int));

	// Populate paged memory arrays
	for ( int idx = 0; idx < totalThreads; idx++ ) {
    	threadCountList[idx] = idx; 
		randNumList[idx] = rand() % 4;
   	}

  	// Populate pinned memory arrays
	memcpy(pinned_threadCountList, threadCountList, totalThreads * sizeof(int));  
	memcpy(pinned_randNumList, randNumList, totalThreads * sizeof(int));
	
	// Show generated values
	//printArray("Thread Count List", threadCountList, totalThreads);
	//printArray("Random Number List", randNumList, totalThreads);
	
	// Run and time operations using paged memory
	//printf("\nPaged Memorry\n");
	start = clock();
	runOperations(numBlocks, totalThreads, threadCountList, randNumList);
	end = clock();
	timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("\nPaged Memory Time: %f\n", timePassedMiliSeconds);

	// Run and time operations using paged memory
	//printf("\nPinned Memorry\n");
	start = clock();
	runOperations(numBlocks, totalThreads, pinned_threadCountList, pinned_randNumList);
	end = clock();
	timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("\nPinned Memory Time: %f\n", timePassedMiliSeconds);

	cudaFreeHost(pinned_threadCountList);
	cudaFreeHost(pinned_randNumList);

	printf("\nEND\n");
	
	return 0;
}
