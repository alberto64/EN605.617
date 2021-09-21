//Based on the work of Andrew Krepps
#include <stdio.h>

struct ThreadVariables {
	int threadCountList[];
 	int randNumList[];
	int resultList[];
};

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = 64;
	int blockSize = 1;
	
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

	// Initialize operation arrays
	pthread_t threads[arraySize];
	int threadCountList[arraySize];
 	int randNumList[arraySize];
	int resultList[arraySize];

	struct ThreadVariables *cpuVar;
    cpuVar = malloc(sizeof(struct threadVariables));

	(*cpuVar).threadCountList = threadCountList;
	(*cpuVar).randNumList = randNumList;
	(*cpuVar).resultList = resultList;

   	// Populate elements of both arrays          
   	for ( int idx = 0; idx < arraySize; idx++ ) {
    	threadCountList[idx] = idx; 
		randNumList[idx] = rand() % 4;
   	}

	// Test using gpu threads
	int *dev_threadCountList, *dev_randNumList, *dev_c;
	
	cudaMalloc((void**)&dev_threadCountList, arraySize * sizeof(int));
	
	cudaMalloc((void**)&dev_randNumList, arraySize * sizeof(int));
	
	cudaMalloc((void**)&dev_resultList, arraySize * sizeof(int));

	cudaMemcpy(dev_threadCountList, threadCountList, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_randNumList, randNumList, arraySize * sizeof(int), cudaMemcpyHostToDevice);

	auto start = std::chrono::high_resolution_clock::now();
	
	addCUDA<<<numBlocks,blockSize>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	subCUDA<<<numBlocks,blockSize>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	multCUDA<<<numBlocks,blockSize>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	modCUDA<<<numBlocks,blockSize>>> (dev_threadCountList, dev_randNumList, dev_resultList);
	
	auto stop = std::chrono::high_resolution_clock::now();

	cudaFree(dev_threadCountList);
	
	cudaFree(dev_randNumList);
	
	cudaFree(dev_resultList);

	std::cout <endl<< " Time elapsed on GPU: " << std::chrono::duration_castchrono::nanoseconds>(stop - start).count() << "ns\n";

	// Test using cpu threads
	auto startCpu = std::chrono::high_resolution_clock::now();


	for(int idx = 0; idx < arraySize ; idx++) {
		thread = pthread_create(&threads[idx], NULL, cpuMain, (void *)cpuVar)
		if (thread) {
			printf("Error:unable to create thread, %d\n", thread);
			exit(-1);
		}
	}

	auto stopCpu = std::chrono::high_resolution_clock::now();

	std::cout << " Time elapsed on CPU: " << std::chrono::duration_castchrono::nanoseconds>(stopCpu - startCpu).count() << "ns\n";

	return 0;
}

void cpuMain(void* threadVariables) {
	
	// Initialize operation arrays
	struct ThreadVariables *cpuVariables = (struct ThreadVariables*)threadVariables;
	thread_idx = pthread_getthreadid_np();

	add((*cpuVariables).threadCountList, (*cpuVariables).randNumList , (*cpuVariables).resultList , (int) thread_idx);
	sub((*cpuVariables).threadCountList, (*cpuVariables).randNumList , (*cpuVariables).resultList , (int) thread_idx);
	mult((*cpuVariables).threadCountList, (*cpuVariables).randNumList , (*cpuVariables).resultList , (int) thread_idx);
	mod((*cpuVariables).threadCountList, (*cpuVariables).randNumList , (*cpuVariables).resultList , (int) thread_idx);
}

void addCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	resultList[idx] = threadCountList[idx] + randNumList[idx]; 
}

void subCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	resultList[idx] = threadCountList[idx] - randNumList[idx]; 
}

void multCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	resultList[idx] = threadCountList[idx] * randNumList[idx]; 
}

void modCUDA(int *threadCountList, int *randNumList, int *resultList) { 
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	resultList[idx] = threadCountList[idx] % randNumList[idx]; 
}

void add(int *threadCountList, int *randNumList, int *resultList, int thread_idx) { 
	resultList[thread_idx] = threadCountList[thread_idx] + randNumList[thread_idx]; 
}

void sub(int *threadCountList, int *randNumList, int *resultList, int thread_idx) { 
	resultList[thread_idx] = threadCountList[thread_idx] - randNumList[thread_idx]; 
}

void mult(int *threadCountList, int *randNumList, int *resultList, int thread_idx) { 
	resultList[thread_idx] = threadCountList[thread_idx] * randNumList[thread_idx]; 
}

void mod(int *threadCountList, int *randNumList, int *resultList, int thread_idx) { 
	resultList[thread_idx] = threadCountList[thread_idx] % randNumList[thread_idx]; 
}
