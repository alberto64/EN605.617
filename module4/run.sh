nvcc operations.cu -o operations.sh
./operations.sh 64 1
./operations.sh 256 4
./operations.sh 
./operations.sh 6240 520
./operations.sh 1000000 5000

nvcc ceasarCipher.cu -o ceasarCipher.sh
./ceasarCipher.sh "This is a message"
./ceasarCipher.sh "This is a message" 5

