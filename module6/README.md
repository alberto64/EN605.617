Run `./run.sh` to start a set of tests to compare timing between using regular global memory and register memory.

Making the GPU use registers dirrectly intead of global memory makes it so that memory in use doesn't have to go throught different levels of memory and it can stay close to the GPU using the GPU's register. 