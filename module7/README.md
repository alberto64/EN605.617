Run `./run.sh` to start a set of tests to compare timing using different array sizes.

Comparing the times we have been obtaining with different memory types using we can say that using pinned memory with shared or register memory is very fast, but using streams with pinned memory beats those by a lot which demonstrates how powerfull multiple concurency is. 

In the stretch problem we were able to demonstrate better how matched register and shared memory are, Although register memory is quicker, shared memory is pretty close behind and its understandable why since they both are so close to the GPU.