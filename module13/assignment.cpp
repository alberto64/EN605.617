//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16
#define NUM_SUB_BUFFER_ELEMENTS 4

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    int * inputOutput;
    char** FILTER_LIST[3] = {"average", "square", "cube"};

    int platform = DEFAULT_PLATFORM; 
    bool useMap  = DEFAULT_USE_MAP;

    std::cout << "Simple buffer and sub-buffer Example" << std::endl;

    int inputModulator = 1;
    if (argc >= 2) {
		inputModulator = atoi(argv[1]);
	}


    // First, select an OpenCL platform to run on.  
    clGetPlatformIDs(0, NULL, &numPlatforms);

 
    platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);

    clGetPlatformIDs(numPlatforms, platformIDs, NULL);

    std::ifstream srcFile("assignment.cl");

    std::string srcProg(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(platformIDs[platform], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");

    clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);    

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, numDevices, &deviceIDs[0], NULL);

    cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[platform], 0};

    context = clCreateContext(contextProperties, numDevices, deviceIDs, NULL, NULL, &errNum);

    // Create program from source
    program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);

    // Build program
    errNum = clBuildProgram(program, numDevices, deviceIDs, "-I.", NULL, NULL);

    // create buffers and sub-buffers
    inputOutput = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++) {
        inputOutput[i] = i * inputModulator;
    }

    // create a single buffer to cover all the input data
    cl_mem main_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices, NULL, &errNum);

    // now for all devices other than the first create a sub-buffer
    for (unsigned int i = 0; i < numDevices; i++) {
        cl_buffer_region region = {NUM_SUB_BUFFER_ELEMENTS * i * sizeof(int), NUM_BUFFER_ELEMENTS * sizeof(int)};
        cl_mem buffer = clCreateSubBuffer(main_buffer,
            CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
            &region, &errNum);
        
        buffers.push_back(buffer);
    }

    // Create command queues
    for (unsigned int i = 0; i < numDevices; i++)
    {
        InfoDevice<cl_device_type>::display(deviceIDs[i], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");

        cl_command_queue queue = clCreateCommandQueue(context, deviceIDs[i], 0, &errNum);

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(program, &FILTER_LIST[inputModulator % 3], &errNum);

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);

        kernels.push_back(kernel);
    }

    if (useMap) 
    {
        cl_int * mapPtr = (cl_int*) clEnqueueMapBuffer(
            queues[numDevices - 1],
            main_buffer,
            CL_TRUE,
            CL_MAP_WRITE,
            0,
            sizeof(cl_int) * NUM_BUFFER_ELEMENTS * numDevices,
            0,
            NULL,
            NULL,
            &errNum);
        checkErr(errNum, "clEnqueueMapBuffer(..)");

        for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
        {
            mapPtr[i] = inputOutput[i];
        }

        errNum = clEnqueueUnmapMemObject(
            queues[numDevices - 1],
            main_buffer,
            mapPtr,
            0,
            NULL,
            NULL);
        checkErr(errNum, "clEnqueueUnmapMemObject(..)");
    }
    else 
    {
        // Write input data
        errNum = clEnqueueWriteBuffer(
            queues[numDevices - 1],
            main_buffer,
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput,
            0,
            NULL,
            NULL);
    }

    std::vector<cl_event> events;
    // call kernel for each device
    for (unsigned int i = 0; i < queues.size(); i++)
    {
        cl_event event;

        size_t gWI = NUM_BUFFER_ELEMENTS;

        errNum = clEnqueueNDRangeKernel(
            queues[i], 
            kernels[i], 
            1, 
            NULL,
            (const size_t*)&gWI, 
            (const size_t*)NULL, 
            0, 
            0, 
            &event);

        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);

    if (useMap)
    {
        cl_int * mapPtr = (cl_int*) clEnqueueMapBuffer(
            queues[numDevices - 1],
            main_buffer,
            CL_TRUE,
            CL_MAP_READ,
            0,
            sizeof(cl_int) * NUM_BUFFER_ELEMENTS * numDevices,
            0,
            NULL,
            NULL,
            &errNum);

        for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
        {
            inputOutput[i] = mapPtr[i];
        }

        errNum = clEnqueueUnmapMemObject(
            queues[numDevices - 1],
            main_buffer,
            mapPtr,
            0,
            NULL,
            NULL);

        clFinish(queues[numDevices - 1]);
    }
    else 
    {
        // Read back computed data
        clEnqueueReadBuffer(queues[numDevices - 1],
            main_buffer,
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput,
            0,
            NULL,
            NULL);
    }

    // Display output in rows
    for (unsigned i = 0; i < numDevices; i++)
    {
        for (unsigned elems = i * NUM_BUFFER_ELEMENTS; elems < ((i+1) * NUM_BUFFER_ELEMENTS); elems++)
        {
            std::cout << " " << inputOutput[elems];
        }

        std::cout << std::endl;
    }

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
