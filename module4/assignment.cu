#include <stdio.h>
#include <stdlib.h>

__global__ void addCUDA(const int *threadCountList, const int *randNumList, int *resultList) { 
	unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] + randNumList[idx]; 
}

__global__ void subCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	resultList[idx] = threadCountList[idx] - randNumList[idx]; 
}

__global__ void multCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	resultList[idx] = threadCountList[idx] * randNumList[idx]; 
}

__global__ void modCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	resultList[idx] = threadCountList[idx] % randNumList[idx]; 
}

void printArray(const char* name, int *array, int size) {
	printf("\n%s: [ ", name);
	for(int idx = 0; idx < size; idx++) {
		printf("%i, ", array[idx]);
	}
	printf("]");
}

int main(int argc, char** argv)
{
	// Based on the work of Andrew Krepps
	// read command line arguments
	int totalThreads = 64;
	int blockSize = 4;
	
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

	int arraySize = totalThreads;
	int threadCountList[arraySize];
	int randNumList[arraySize];
	int addresultList[arraySize];
	// int subresultList[arraySize];
	// int multresultList[arraySize];
	// int modresultList[arraySize];

	for ( int idx = 0; idx < arraySize; idx++ ) {
    	threadCountList[idx] = idx; 
		randNumList[idx] = rand() % 4;
   	}

	printArray("Thread Count List", threadCountList, arraySize);
	printArray("Random Number List", randNumList, arraySize);
	
	int *dev_threadCountList, *dev_randNumList, *dev_resultList;

	cudaMalloc((void**)&dev_threadCountList, arraySize * sizeof(int));
	
	cudaMalloc((void**)&dev_randNumList, arraySize * sizeof(int));
	
	cudaMalloc((void**)&dev_resultList, arraySize * sizeof(int));

	cudaMemcpy(dev_threadCountList, threadCountList, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_randNumList, randNumList, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	
	addCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);

	cudaMemcpy(addresultList, dev_resultList, arraySize * sizeof(int), cudaMemcpyDeviceToHost); 

	// subCUDA<<<numBlocks,blockSize>>> (dev_threadCountList, dev_randNumList, dev_resultList);

	// cudaMemcpy(subresultList, dev_resultList, arraySize * sizeof(int), cudaMemcpyDeviceToHost); 

	// multCUDA<<<numBlocks,blockSize>>> (dev_threadCountList, dev_randNumList, dev_resultList);

	// cudaMemcpy(multresultList, dev_resultList, arraySize * sizeof(int), cudaMemcpyDeviceToHost); 

	// modCUDA<<<numBlocks,blockSize>>> (dev_threadCountList, dev_randNumList, dev_resultList);

	// cudaMemcpy(modresultList, dev_resultList, arraySize * sizeof(int), cudaMemcpyDeviceToHost); 

	printArray("Add Result", addresultList, arraySize);
	// printArray("Sub Result", subresultList, arraySize);
	// printArray("MUlt Result", multresultList, arraySize);
	// printArray("Mod Result", modresultList, arraySize);

	
	cudaFree(dev_threadCountList);

	cudaFree(dev_randNumList);
	
	cudaFree(dev_resultList);

	printf("\nEND");
	
	return 0;
}
