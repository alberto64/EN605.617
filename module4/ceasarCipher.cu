#include <stdio.h>
#include <time.h>

/**
* ceasarCypherEncryptCUDA: Given a cipher key and a char array it encrypts it using a ceasar cypher and 
* saves it into a new char array.
*/
__global__ void ceasarCypherEncryptCUDA(int cypherKey, char *stringToEncrypt, char *resultString) { 
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	resultString[idx] = stringToEncrypt[idx] + cypherKey; 
}


/**
* printArray: A method that takes in an a label and an array with its size and it feeds it to printf.
*/
void printWord(const char* name, char *array, int size) {
	printf("\n%s: ", name);
	for(int idx = 0; idx < size; idx++) {
		printf("%c", array[idx]);
	}
	printf("\n");
}

/**
* runCypher: Taking the number of blocks and threads it does a cypher operation and it prints the result. 
* Uses paged memory
*/
void runCypher(int numBlocks, int totalThreads, int cipherKey, char* word) { 
	
	// Prepare result array variables
	char* encryptedWord = (char*) malloc(totalThreads * sizeof(char));

	
	// Prepare cuda variables
	char* dev_word, *dev_encryptedWord;
	cudaMalloc((void**)&dev_word, totalThreads * sizeof(char));
	cudaMalloc((void**)&dev_encryptedWord, totalThreads * sizeof(char));

	// Copy inputs into device memory 
	cudaMemcpy(dev_word, word, totalThreads * sizeof(char), cudaMemcpyHostToDevice);
	
	// Execute each operation and bring result from device to host
	ceasarCypherEncryptCUDA<<<numBlocks,totalThreads>>> (cipherKey, dev_word, dev_encryptedWord);
	cudaMemcpy(encryptedWord, dev_encryptedWord, totalThreads * sizeof(int), cudaMemcpyDeviceToHost); 

	// Turned of to minimize printing
	printWord("Paged Word", encryptedWord, totalThreads);

	
	// Free reserved memory
	cudaFree(dev_word);
	cudaFree(dev_encryptedWord);
	free(encryptedWord);
}

/**
* runCypherOnHost: Taking the number of blocks and threads it does a cypher operation and it prints the result. 
* Uses pinned memory.
*/
void runCypherOnHost(int numBlocks, int totalThreads, int cipherKey, char* word) { 
	
	// Prepare result array variables
	char* encryptedWord = (char*) malloc(totalThreads * sizeof(char));

	cudaMallocHost((void**)&encryptedWord, totalThreads * sizeof(char));


	// Prepare cuda variables
	char* dev_word, *dev_encryptedWord;

	// Copy inputs into device memory
	cudaHostGetDevicePointer(&dev_word, word, 0);
	cudaHostGetDevicePointer(&dev_encryptedWord, encryptedWord, 0);
	
	// Execute each operation and bring result from device to host
	ceasarCypherEncryptCUDA<<<numBlocks,totalThreads>>> (cipherKey, dev_word, dev_encryptedWord);

	// Synchonize data between device and host
	cudaDeviceSynchronize();

	// Turned of to minimize printing
	printWord("Pinned Word", encryptedWord, totalThreads);

	
	// Free reserved memory
	cudaFree(dev_word);
	cudaFree(dev_encryptedWord);
	cudaFreeHost(encryptedWord);
}

int main(int argc, char** argv)
{
	// Based on the work of Andrew Krepps
	
	// Set default values in case arguments don't come in command line.
	char* word = (char*) "Hello World";
	int key = 3;
	// read command line arguments
	if (argc >= 2) {
		word = (char*) argv[1];
	}
	if (argc >= 3) {
		key = atoi(argv[2]);
	}

	int totalThreads = sizeof(word);
	int numBlocks = 1;
	
	printf("Total Threads: %d\nBlock Count: %d\n, Word: %s\n", totalThreads, numBlocks, word);

	// Set up variables for timing
	clock_t start, end;
	double timePassedMiliSeconds;
	
	// Set up pinned memory space
	char* pinned_word;
	cudaMallocHost((void**)&pinned_word, totalThreads * sizeof(char));

  	// Populate pinned memory arrays
	memcpy(pinned_word, word, totalThreads * sizeof(char));  
	
	// Run and time operations using paged memory
	start = clock();
	runCypher(numBlocks, totalThreads, key, word);
	end = clock();
	timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("\nPaged Memory Time: %f Miliseconds\n", timePassedMiliSeconds);

	// Run and time operations using paged memory
	start = clock();
	runCypherOnHost(numBlocks, totalThreads, key, pinned_word);
	end = clock();
	timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("\nPinned Memory Time: %f Miliseconds\n", timePassedMiliSeconds);

	// Free reserved memory
	cudaFreeHost(pinned_word);
	
	return 0;
}
