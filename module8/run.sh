nvcc matrixOperation.cu -lcublas -o matrix.sh
nvcc solvingOperation.cu -lcublas -lcusolver -o solver.sh

./matrix.sh 4
./matrix.sh 64
./matrix.sh 264
./matrix.sh 

./solver.sh 4
./solver.sh 16
./solver.sh 32
./solver.sh 64
