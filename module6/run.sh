nvcc pagedOperations.cu -o paged.sh
nvcc registerOperations.cu -o register.sh
./paged.sh 64 1
./register.sh 64 1

./paged.sh 256 4
./register.sh 256 4

./paged.sh 
./register.sh 

./paged.sh 6240 520
./register.sh 6240 520

