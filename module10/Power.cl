
__kernel void power(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);
    int count = 0;
    result[gid] = 1;
    while(count < b[gid]) 
    {
        result[gid] = result[gid] * a[gid];
        count = count + 1;
    }
}
