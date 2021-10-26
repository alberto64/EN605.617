nvcc matrixOperation.cu -lcublas -o matrix.sh
nvcc solvingOperation.cu -lcublas -lcusolver -o solver.sh

echo "Matrix Calculations.\n"

./matrix.sh 4
./matrix.sh 64
./matrix.sh 264
./matrix.sh 

echo "Solver Calculations.\n"

./solver.sh 4
./solver.sh 16
./solver.sh 32
./solver.sh
