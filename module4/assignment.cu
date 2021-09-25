#include <stdio.h>

__global__ void addCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] + randNumList[idx]; 
}

__global__ void subCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] - randNumList[idx]; 
}

__global__ void multCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultList[idx] = threadCountList[idx] * randNumList[idx]; 
}

__global__ void modCUDA(int *threadCountList, int *randNumList, int *resultList) { 
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

void printArray(const char* name, int *array, int size) {
	printf("\n%s: [ ", name);
	for(int idx = 0; idx < size; idx++) {
		printf("%i ", array[idx]);
	}
	printf("]");
}

void pagedMemoryOperations(int numBlocks, int totalThreads, int* threadCountList, int* randNumList) { 
	
	int addresultList[totalThreads];
	int subresultList[totalThreads];
	int multresultList[totalThreads];
	int modresultList[totalThreads];
	int *dev_threadCountList, *dev_randNumList, *dev_resultList;

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

	printArray("Add Result", addresultList, totalThreads);
	printArray("Sub Result", subresultList, totalThreads);
	printArray("Mult Result", multresultList, totalThreads);
	printArray("Mod Result", modresultList, totalThreads);
	
	cudaFree(dev_threadCountList);

	cudaFree(dev_randNumList);
	
	cudaFree(dev_resultList);
}

void pinnedMemoryOperations(int numBlocks, int totalThreads, int* threadCountList, int* randNumList) { 
	
	int addresultList[totalThreads];
	int subresultList[totalThreads];
	int multresultList[totalThreads];
	int modresultList[totalThreads];
	int *dev_threadCountList, *dev_randNumList, *dev_addList, *dev_resultList;

	cudaMallocHost((void**)&dev_threadCountList, totalThreads * sizeof(int));
	cudaMallocHost((void**)&dev_randNumList, totalThreads * sizeof(int));
	cudaMallocHost((void**)&dev_addList, totalThreads * sizeof(int));
	cudaMallocHost((void**)&dev_resultList, totalThreads * sizeof(int));

	memcpy(dev_threadCountList, threadCountList, totalThreads * sizeof(int));
	memcpy(dev_randNumList, randNumList, totalThreads * sizeof(int));
	
	addCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_addList);
	memcpy(addresultList, dev_addList, totalThreads * sizeof(int)); 

	subCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	memcpy(subresultList, dev_resultList, totalThreads * sizeof(int)); 

	multCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	memcpy(multresultList, dev_resultList, totalThreads * sizeof(int)); 

	modCUDA<<<numBlocks,totalThreads>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	memcpy(modresultList, dev_resultList, totalThreads * sizeof(int)); 

	printArray("Add Result", addresultList, totalThreads);
	printArray("Sub Result", subresultList, totalThreads);
	printArray("Mult Result", multresultList, totalThreads);
	printArray("Mod Result", modresultList, totalThreads);
	
	cudaFreeHost(dev_threadCountList);

	cudaFreeHost(dev_randNumList);
	
	cudaFreeHost(dev_resultList);
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

	int threadCountList[totalThreads];
	int randNumList[totalThreads];

	for ( int idx = 0; idx < totalThreads; idx++ ) {
    	threadCountList[idx] = idx; 
		randNumList[idx] = rand() % 4;
   	}

	printArray("Thread Count List", threadCountList, totalThreads);
	printArray("Random Number List", randNumList, totalThreads);
	
	printf("\nPAGED\n");
	pagedMemoryOperations(numBlocks, totalThreads, threadCountList, randNumList);
	printf("\nPINNED\n");
	pinnedMemoryOperations(numBlocks, totalThreads, threadCountList, randNumList);

	printf("\nEND");
	
	return 0;
}
