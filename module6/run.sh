nvcc operationsConst.cu -o constant.sh
nvcc operationsShared.cu -o shared.sh
./constant.sh 64 1
./shared.sh 64 1

./constant.sh 256 4
./shared.sh 256 4

./constant.sh 
./shared.sh 

./constant.sh 6240 520
./shared.sh 6240 520

