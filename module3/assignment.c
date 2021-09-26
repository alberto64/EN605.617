//Based on the work of Andrew Krepps
#include <stdio.h>
#include <pthread.h>

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

	int arraySize = totalThreads*blockSize;

	// Initialize operation arrays
	pthread_t threads[arraySize];
	int blockThreadList[arraySize];
 	int rand3NumList[arraySize];

   	// Populate elements of both arrays          
   	for ( int idx = 0; idx < arraySize; idx++ ) {
    	blockThreadList[idx] = idx; 
		rand3NumList[idx] = rand() % 4;
   	}
}

int* add (int* threadList, int* randList, int arraySize) {
	int sumResultList[arraySize];
	for(int idx = 0; idx < arraySize ; idx++) {
		sumResultList[idx] = threadList[idx] + randList[idx];
	}
	return sumResultList;
}

int* sub (int* threadList, int* randList, int arraySize) {
	int subResultList[arraySize];
	for(int idx = 0; idx < arraySize ; idx++) {
		subResultList[idx] = threadList[idx] - randList[idx];
	}
	return subResultList;
}

int* mult (int* threadList, int* randList, int arraySize) {
	int multResultList[arraySize];
	for(int idx = 0; idx < arraySize ; idx++) {
		multResultList[idx] = threadList[idx] * randList[idx];
	}
	return multResultList;
}

int* mod (int* threadList, int* randList, int arraySize) {
	int modResultList[arraySize];
	for(int idx = 0; idx < arraySize ; idx++) {
		modResultList[idx] = threadList[idx] % randList[idx];
	}
	return modResultList;
}