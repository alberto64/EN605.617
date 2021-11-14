//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
using namespace std;
using namespace std::chrono;

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

///
//	For Convoloution example
//
void convolution(const unsigned int inputSignalSize, const unsigned int filterSize, const unsigned int outputSignalSize)
{
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

	cl_uint inputSignal[inputSignalSize][inputSignalSize];
	for(int i = 0; i < inputSignalSize; i++) {
		for(int j = 0; j < inputSignalSize; j++) {
			inputSignal[i][j] = rand() % 9;
		}
	}

	cl_uint outputSignal[outputSignalSize][outputSignalSize];

	cl_uint mask[filterSize][filterSize];
	for(int i = 0; i < filterSize; i++) {
		for(int j = 0; j < filterSize; j++) {
			mask[i][j] = rand() % 1;
		}
	}

    // First, select an OpenCL platform to run on.  
	clGetPlatformIDs(0, NULL, &numPlatforms);
 
	platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);

    clGetPlatformIDs(numPlatforms, platformIDs, NULL);


	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
	    if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			break;
	   }
	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);

	std::ifstream srcFile("Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalSize * inputSignalSize,
		static_cast<void *>(inputSignal),
		&errNum);

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * filterSize * filterSize,
		static_cast<void *>(mask),
		&errNum);

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalSize * outputSignalSize,
		NULL,
		&errNum);

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalSize);
	clSetKernelArg(kernel, 4, sizeof(cl_uint), &filterSize);

	const size_t globalWorkSize[2] = { outputSignalSize, outputSignalSize };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		NULL);
    
	clEnqueueReadBuffer(
		queue, 
		outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_uint) * outputSignalSize * outputSignalSize, 
		outputSignal,
        0, 
		NULL, 
		NULL);

    // Output the result buffer
    for (int y = 0; y < outputSignalSize; y++)
	{
		for (int x = 0; x < outputSignalSize; x++)
		{
			std::cout << outputSignal[y][x] << " ";
		}
		std::cout << std::endl;
	}

    std::cout << std::endl << "Executed program succesfully." << std::endl;
}

int main(int argc, char** argv) 
{
	    // Based on the work of Andrew Krepps
	// Set default values in case arguments don't come in command line.
	int inputSignalSize = 49;
	int filterSize = 7;

	// read command line arguments
	if (argc >= 2) {
		inputSignalSize = atoi(argv[1]);
	}
	if (argc >= 3) {
		filterSize = atoi(argv[2]);
	}

	int outputSignalSize = inputSignalSize - filterSize + 1;

    cout << "Total Input Signal Size: " << inputSignalSize << "x" << inputSignalSize << endl;
    cout << "Total Filter Size: " << filterSize << "x" << filterSize << endl;
    cout << "Total Output Size: " << outputSignalSize << "x" << outputSignalSize << endl;

    auto start = high_resolution_clock::now();
	convolution(inputSignalSize, filterSize, outputSignalSize);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Time taken by operations: " << duration.count() << " milliseconds" << endl;
	return 0;
}
