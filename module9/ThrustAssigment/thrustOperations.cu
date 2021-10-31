#include <stdio.h>
#include <time.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/fill.h>

/**
* add: A method that add two arrays and places the result in a third array using thrust
*/
void add(thrust::device_vector<int>& threadCountList, thrust::device_vector<int>& randNumList, thrust::device_vector<int>& resultList) { 
    thrust::transform(threadCountList.begin(), threadCountList.end(), randNumList.begin(), resultList.begin(), thrust::plus<int>());
}

/**
* sub: A method that substract two arrays and places the result in a third array using thrust
*/
void sub(thrust::device_vector<int>& threadCountList, thrust::device_vector<int>& randNumList, thrust::device_vector<int>& resultList) { 
    thrust::transform(threadCountList.begin(), threadCountList.end(), randNumList.begin(), resultList.begin(), thrust::minus<int>());
}

/**
* mult: A method that multiplies two arrays and places the result in a third array using thrust
*/
void mult(thrust::device_vector<int>& threadCountList, thrust::device_vector<int>& randNumList, thrust::device_vector<int>& resultList) { 
    thrust::transform(threadCountList.begin(), threadCountList.end(), randNumList.begin(), resultList.begin(), thrust::multiplies<int>());
}

/**
* mod: A method that does the modulus between two arrays and places the result in a third thrust
*/
void mod(thrust::device_vector<int>& threadCountList, thrust::device_vector<int>& randNumList, thrust::device_vector<int>& resultList) { 
    thrust::transform(threadCountList.begin(), threadCountList.end(), randNumList.begin(), resultList.begin(), thrust::modulus<int>());
}

/**
* printArray: A method that takes in an a label and an array with its size and it feeds it to printf.
*/
void printArray(const char* name, thrust::host_vector<int>& array, int size) {
	printf("\n%s: [ ", name);
	for (int idx = 0; idx < size; idx++) {
		printf("%i ", array[idx]);
	}
	printf("]\n");
}

/**
* runOperations: Taking the number of blocks and threads it does 4 operations on the two 
* given arrays and prints their results.
*/
void runOperations(int vectorLength) { 
    
	// Setup Timing Variables
	cudaEvent_t start, stop; 
	float elapsedTimeInMiliseconds; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
  
	// Setup host memory variables
    thrust::host_vector<int> host_threadCountList(vectorLength);
	thrust::host_vector<int> host_randNumberList(vectorLength);

	// Populate host inputs
	for(int idx = 0; idx < vectorLength; idx++) { 
		host_threadCountList[idx] = idx; 
		host_randNumberList[idx] = rand() % 4; 
	} 

	// Setup device memory variables
	thrust::device_vector<int> device_threadCountList = host_threadCountList;
	thrust::device_vector<int> device_randNumberList = host_randNumberList;
	thrust::device_vector<int> device_resultList(vectorLength);

	// Turned of to minimize printing
	// printArray("Thread Count", host_threadCountList, vectorLength);
	// printArray("Random Numbers", host_randNumberList, vectorLength);

	// Start event
	cudaEventRecord(start);

	// Execute each operation and bring result from device to host
	add(device_threadCountList, device_randNumberList, device_resultList);
	thrust::host_vector<int> host_addResultList = device_resultList;

	sub(device_threadCountList, device_randNumberList, device_resultList);
	thrust::host_vector<int> host_subResultList = device_resultList;
	
	mult(device_threadCountList, device_randNumberList, device_resultList);
	thrust::host_vector<int> host_multResultList = device_resultList;

	mod(device_threadCountList, device_randNumberList, device_resultList);
	thrust::host_vector<int> host_modResultList = device_resultList;

  	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop); 
  	cudaEventElapsedTime(&elapsedTimeInMiliseconds, start, stop); 
	printf("Operations Event Time: %f Miliseconds\n", elapsedTimeInMiliseconds) * 100;

	// Turned of to minimize printing
	// printArray("Add Result", host_addResultList, vectorLength);
	// printArray("Sub Result", host_subResultList, vectorLength);
	// printArray("Mult Result", host_multResultList, vectorLength);
	// printArray("Mod Result", host_modResultList, vectorLength);
}

int main(int argc, char** argv) {
	// Based on the work of Andrew Krepps
	// Set default values in case arguments don't come in command line.
	int vectorLength = 1024;

	// read command line arguments
	if (argc >= 2) {
		vectorLength = atoi(argv[1]);
	}

	printf("\nVector Size: %d\n", vectorLength);
	
	runOperations(vectorLength);

	return 0;
}
