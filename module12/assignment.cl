//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void average(__global * buffer, __global int size)
{
	size_t id = get_global_id(0);
	int sum = 0;
	for(int i = 0; i < size; i++) {
		sum = buffer[i];
	}
	buffer[id] = sum/size;
}