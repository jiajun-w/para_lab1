#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "gen_matrix.h"
#include "my_malloc.h"
//#include "gemm.h"
/*************************************************/
/*matrix matrix multiplication, need optimization*/
/*************************************************/
// C = A * B, A(M*K) B(K*N) C(M*N)
void gemm(double *A, double *B, double *C, int M, int N, int K); 
void mm(double *A, double *B, double *C, int M, int N, int K){
    int x, y, k;
    for(y=0; y<M;y++)
        for(x=0; x<N; x++){
            C[x + y*N] = 0;
            for(k=0;k<K; k++)
                C[x + y*N] += A[k + y*K] * B[k + x*K];
        } 
}

/*************************************************/
/*print out matrix, need to be replaced with ours*/
/*************************************************/

void print_matrix(double *result, int dim_size) {
  int x, y;
  for (y = 0; y < dim_size; ++y) {
    for (x = 0; x < dim_size; ++x) {
      printf("%f ", result[y * dim_size + x]);
    }
    printf("\n");
  }
  printf("\n");
}

/*************************************************/
/*********************main************************/
/*************************************************/

int main(int argc, char** argv){

    if (argc != 4) {
        printf("usage: debug_perf test_set matrix_dimension_size\n");
        exit(1);
    }
    int debug_perf = atoi(argv[1]);
    int test_set = atoi(argv[2]);
    int matrix_dimension_size = atoi(argv[3]);
    int num_arg_matrices = init_gen_sub_matrix(test_set);


    int rank,numtasks;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
//    printf("number of tasks = %d\n",numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);


/**************************************************/
/**********allocate receive buffer*****************/
/**************************************************/
    double **buffer = (double **)my_malloc(sizeof(double *) * 2);
    int M = matrix_dimension_size / numtasks;
    int N =  matrix_dimension_size / numtasks / 2;
    int K = matrix_dimension_size;
//    printf("M=%d, N=%d, K=%d\n",M,N,K);
    buffer[0] =  (double*)my_malloc(sizeof(double)* K * N);//buffer[0] always store first half of matrix B, buffer[1] always store second half
    buffer[1] =  (double*)my_malloc(sizeof(double)* K * N);
    //allocate result matrices
    double *result =  (double*)my_malloc(sizeof(double)*M*K);


/**************************************************/
/****allocate matrices and get sub matrices *******/
/**************************************************/
    double **matrix = (double **)my_malloc(sizeof(double *) * num_arg_matrices);
    int x_lo, x_hi, x_stride, y_lo, y_hi, y_stride, row_major;
    //first matrix is in row major
    y_lo = M * rank;
    y_hi = y_lo + M - 1;
    x_lo = 0;
    x_hi = K - 1;
    x_stride = 1;
    y_stride = 1;
    row_major = 1; //row_major
    matrix[0] = (double*)my_malloc(sizeof(double) * M * K);
    if (gen_sub_matrix(rank, test_set, 0, matrix[0], x_lo, x_hi, x_stride, y_lo, y_hi, y_stride, row_major) == NULL ) {
        printf("inconsistency in gen_sub_matrix\n");
        exit(1);
    }

    //rest matrices are in column major
    //matrix, each process get 1/numtasks of matric i
    for(int i=1; i<num_arg_matrices; i++){
        matrix[i] = (double*)my_malloc(sizeof(double)* M * K);
        x_lo = M * rank;
        x_hi = x_lo + M - 1;
        y_lo = 0;
        y_hi = K - 1;
        x_stride = 1;
        y_stride = 1;
        row_major = 0; //column_major
        if (gen_sub_matrix(rank, test_set, i, matrix[i], x_lo, x_hi, x_stride, y_lo, y_hi, y_stride, row_major) == NULL) {
            printf("inconsistency in gen_sub_matrix\n");
            exit(1);
        }
    }  

    MPI_Status status;
    MPI_Request request = MPI_REQUEST_NULL;
    int result_turn = 0;
    double *res[2];
    res[0] = result;
    res[1] = matrix[0];


/**************************************************/
/**************compute matrices *******************/
/**************************************************/    
    //a serial way of doing matrices multiplication
    double starttime = MPI_Wtime();
    for(int multiplier=1; multiplier<num_arg_matrices; multiplier++){

        double *A, *B, *C; // C=A*B
        if(multiplier == 1){
            A = matrix[0];
        }
        else{
            A = res[result_turn^0x1];
        }
        C = res[result_turn];
        
        //first iteration, only load, blocking broadcast
        //process 0 first broadcast its block
        int buffer_turn = 0;
        int root = 0;
        if(rank == root)
            buffer[buffer_turn] = matrix[multiplier] + buffer_turn * (K*N);
        MPI_Bcast(buffer[buffer_turn], K*N, MPI_DOUBLE, root, MPI_COMM_WORLD);

//        if(rank == 0)        
//        printf("prcess 0 send M%d  1st col = %f %f %f %f %f %f %f %f  \n",multiplier, matrix[multiplier][0],matrix[multiplier][1],matrix[multiplier][2],matrix[multiplier][3],matrix[multiplier][4],matrix[multiplier][5],matrix[multiplier][6],matrix[multiplier][7]);
//        if(rank == 0)        
//        printf("prcess 0 __send M%d  1st col = %f %f %f %f %f %f %f %f  \n",rank, multiplier, buffer[buffer_turn][0],buffer[buffer_turn][1],buffer[buffer_turn][2],buffer[buffer_turn][3],buffer[buffer_turn][4],buffer[buffer_turn][5],buffer[buffer_turn][6],buffer[buffer_turn][7]);
        

        // intermediate iterations, double buffering to hide load latency, 
        for(int i=1; i<numtasks*2;i++){
            root = i/2; //each processor do seriall non-blocking broadcast
            if(rank == root)
                buffer[buffer_turn^0x1] = matrix[multiplier] + (buffer_turn^0x1) * (K*N) ;
            MPI_Ibcast(buffer[buffer_turn^0x1], K*N, MPI_DOUBLE, root, MPI_COMM_WORLD,&request);
//            if(rank == root)
//                printf("process %d sent Matrix %d, part %d\n", root, multiplier, buffer_turn^0x1);
            B=buffer[buffer_turn];
            //mm(A, B, C, M, N, K); 
            gemm(A, B, C, M, N, K); 
            C += N; 
            buffer_turn = buffer_turn ^ 0x1;
            MPI_Wait(&request, &status);
        }

        //last iteration only computation is required
        B=buffer[buffer_turn];
        //mm(A, B, C, M, N, K); 
        gemm(A, B, C, M, N, K); 
        //end of a full matrix matrix mulipication

        result_turn = result_turn ^ 0x1;
    }

    double endtime = MPI_Wtime();
    double time = endtime - starttime;
    printf("%d processes cost time %f\n",numtasks,time);
/**************************************************/
/**************print result matrices **************/
/**************************************************/   
/*    
    for(int i=0; i<numtasks; i++){
        if(rank == i){
            printf("result at process %d is place at res[%d] %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", i,result_turn, res[result_turn][0],res[result_turn][1],res[result_turn][2],res[result_turn][3],res[result_turn][4],res[result_turn][5],res[result_turn][6],res[result_turn][7],res[result_turn][8], res[result_turn][9], res[result_turn][10], res[result_turn][11], res[result_turn][12], res[result_turn][13], res[result_turn][14], res[result_turn][15]);
        }
    }
*/    
    MPI_Finalize();
    return 0;
}
