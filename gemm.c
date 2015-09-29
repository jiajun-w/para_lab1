#include <stdio.h>
#include <stdlib.h>

#define min( i, j ) ( (i)<(j) ? (i): (j) )
#define KC 256
#define NC 512
/*test only*/
//#define KC 64
//#define NC 64
#define MC 48
#define NR 4
#define MR 4

static inline void rank_1x1(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00;
    c00 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
    }
    c[0] += c00;
    return;
}

static inline void rank_1x2(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01;
    c00 = 0; c01 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    return;
}

static inline void rank_1x3(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c02;
    c00 = 0; c01 = 0; c02 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c02 += a[k] * b[2*K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    c[2]+=c02;
    return;
}

static inline void rank_1x4(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c02, c03;
    c00 = 0; c01 = 0; c02 = 0; c03 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c02 += a[k] * b[2*K+k];
        c02 += a[k] * b[3*K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    c[2]+=c02;
    c[3]+=c02;
    return;
}

static inline void rank_2x1(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c10;
    c00 = 0; c10 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c10 += a[K+k] * b[k];
    }
    c[0] += c00;
    c[N] += c10;
    return;
}

static inline void rank_2x2(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c10, c11;
    c00 = 0; c10 = 0; c01 = 0; c11 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    c[N]+=c10;
    c[N+1]+=c11;
    return;
}

static inline void rank_2x3(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c02, c10, c11, c12;
    c00 = 0; c10 = 0; c01 = 0; c11 = 0; c02 = 0; c12 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c02 += a[k] * b[2*K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
        c12 += a[K+k] * b[2*K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    c[2]+=c02;
    c[N]+=c10;
    c[N+1]+=c11;
    c[N+2]+=c12;
    return;
}

static inline void rank_2x4(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c02, c03, c10, c11, c12, c13;
    c00 = 0; c01 = 0; c02 = 0; c03 = 0; c10 = 0; c11 = 0; c12 = 0; c13 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c02 += a[k] * b[2*K+k];
        c03 += a[k] * b[3*K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
        c12 += a[K+k] * b[2*K+k];
        c13 += a[K+k] * b[3*K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    c[2]+=c02;
    c[3]+=c03;
    c[N]+=c10;
    c[N+1]+=c11;
    c[N+2]+=c12;
    c[N+3]+=c13;
    return;
}

static inline void rank_3x1(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c10, c20;
    c00 = 0; c10 = 0; c20 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c10 += a[K+k] * b[k];
        c20 += a[2*K+k] * b[k];
    }
    c[0] += c00;
    c[N] += c10;
    c[2*N] += c20;
    return;
}

static inline void rank_3x2(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c10, c11 ,c20 ,c21;
    c00 = 0; c01 = 0; c10 = 0; c11 = 0; c20 = 0; c21 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
        c20 += a[2*K+k] * b[k];
        c21 += a[2*K+k] * b[K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    c[N]+=c10;
    c[N+1]+=c11;
    c[2*N]+=c20;
    c[2*N+1]+=c21;
    return;
}

static inline void rank_3x3(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c02, c10, c11, c12, c20, c21, c22;
    c00 = 0; c01 = 0; c02 = 0; c10 = 0; c11 = 0; c12 = 0; c20 = 0; c21 = 0; c22 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c02 += a[k] * b[2*K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
        c12 += a[K+k] * b[2*K+k];
        c20 += a[2*K+k] * b[k];
        c21 += a[2*K+k] * b[K+k];
        c22 += a[2*K+k] * b[2*K+k];

    }
    c[0] += c00;
    c[1]+=c01;
    c[2]+=c02;
    c[N]+=c10;
    c[N+1]+=c11;
    c[N+2]+=c12;
    c[2*N]+=c20;
    c[2*N+1]+=c21;
    c[2*N+2]+=c22;
    return;
}

static inline void rank_3x4(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23;
    c00 = 0; c01 = 0; c02 = 0; c03 = 0; c10 = 0; c11 = 0; c12 = 0; c13 = 0; c20 = 0; c21 = 0; c22 = 0; c23 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c02 += a[k] * b[2*K+k];
        c03 += a[k] * b[3*K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
        c12 += a[K+k] * b[2*K+k];
        c13 += a[K+k] * b[3*K+k];
        c20 += a[2*K+k] * b[k];
        c21 += a[2*K+k] * b[K+k];
        c22 += a[2*K+k] * b[2*K+k];
        c23 += a[2*K+k] * b[3*K+k];

    }
    c[0] += c00;
    c[1]+=c01;
    c[2]+=c02;
    c[3]+=c03;
    c[N]+=c10;
    c[N+1]+=c11;
    c[N+2]+=c12;
    c[N+3]+=c13;
    c[2*N]+=c20;
    c[2*N+1]+=c21;
    c[2*N+2]+=c22;
    c[2*N+3]+=c23;
    return;
}

static inline void rank_4x1(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c10, c20, c30;
    c00 = 0; c10 = 0; c20 = 0; c30 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c10 += a[K+k] * b[k];
        c20 += a[2*K+k] * b[k];
        c30 += a[3*K+k] * b[k];
    }
    c[0] += c00;
    c[N] += c10;
    c[2*N] += c20;
    c[3*N] += c30;
    return;
}

static inline void rank_4x2(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c10, c11 ,c20 ,c21, c30, c31;
    c00 = 0; c10 = 0; c20 = 0; c30 = 0; c01 = 0; c11 = 0; c21 = 0; c31 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
        c20 += a[2*K+k] * b[k];
        c21 += a[2*K+k] * b[K+k];
        c30 += a[3*K+k] * b[k];
        c31 += a[3*K+k] * b[K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    c[N]+=c10;
    c[N+1]+=c11;
    c[2*N]+=c20;
    c[2*N+1]+=c21;
    c[3*N]+=c30;
    c[3*N+1]+=c31;
    return;
}

static inline void rank_4x3(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;
    c00 = 0; c10 = 0; c20 = 0; c30 = 0; c01 = 0; c11 = 0; c21 = 0; c31 = 0; c02 = 0; c12 = 0; c22 = 0; c32 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c02 += a[k] * b[2*K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
        c12 += a[K+k] * b[2*K+k];
        c20 += a[2*K+k] * b[k];
        c21 += a[2*K+k] * b[K+k];
        c22 += a[2*K+k] * b[2*K+k];
        c30 += a[3*K+k] * b[k];
        c31 += a[3*K+k] * b[K+k];
        c32 += a[3*K+k] * b[2*K+k];

    }
    c[0] += c00;
    c[1]+=c01;
    c[2]+=c02;
    c[N]+=c10;
    c[N+1]+=c11;
    c[N+2]+=c12;
    c[2*N]+=c20;
    c[2*N+1]+=c21;
    c[2*N+2]+=c22;
    c[3*N]+=c30;
    c[3*N+1]+=c31;
    c[3*N+2]+=c32;
    return;
}

static inline void rank_4x4(double *a, double *b, double *c , int kc, int M, int N, int K){
    double c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
    c00 = 0; c10 = 0; c20 = 0; c30 = 0; c01 = 0; c11 = 0; c21 = 0; c31 = 0; c02 = 0; c12 = 0; c22 = 0; c32 = 0; c03 = 0; c13 = 0; c23 = 0; c33 = 0;
    for(int k=0; k<kc; k++){
        c00 += a[k] * b[k];
        c01 += a[k] * b[K+k];
        c02 += a[k] * b[2*K+k];
        c03 += a[k] * b[3*K+k];
        c10 += a[K+k] * b[k];
        c11 += a[K+k] * b[K+k];
        c12 += a[K+k] * b[2*K+k];
        c13 += a[K+k] * b[3*K+k];
        c20 += a[2*K+k] * b[k];
        c21 += a[2*K+k] * b[K+k];
        c22 += a[2*K+k] * b[2*K+k];
        c23 += a[2*K+k] * b[3*K+k];
        c30 += a[3*K+k] * b[k];
        c31 += a[3*K+k] * b[K+k];
        c32 += a[3*K+k] * b[2*K+k];
        c33 += a[3*K+k] * b[3*K+k];
    }
    c[0] += c00;
    c[1]+=c01;
    c[2]+=c02;
    c[3]+=c03;
    c[N]+=c10;
    c[N+1]+=c11;
    c[N+2]+=c12;
    c[N+3]+=c13;
    c[2*N]+=c20;
    c[2*N+1]+=c21;
    c[2*N+2]+=c22;
    c[2*N+3]+=c23;
    c[3*N]+=c30;
    c[3*N+1]+=c31;
    c[3*N+2]+=c32;
    c[3*N+3]+=c33;
    return;
}


void gemm(double *A, double *B, double *C, int M, int N, int K){
    //initialize C matrix
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++)
            C[i*N + j] = 0;

    //GEMM with BLIS algorithm
    for(int jc = 0; jc<N; jc+=NC){  //5-th loop
        int nc = min(NC, N-jc);
        double *B_ptr = B + (jc * K);
        double *A_ptr = A;
        double *C_ptr = C + jc;
        for(int rc=0; rc < K; rc += KC){    //4-th loop
            int kc = min(KC, K-rc);
            double *A_ptr4 = A_ptr + rc ;
            double *B_ptr4 = B_ptr + rc ; 
            for(int ic=0; ic < M; ic += MC ){    //3-rd loop
                int mc = min(MC, M-ic);
                double *A_ptr3 = A_ptr4 + ic*K;
                double *C_ptr3 = C_ptr + ic*N;
                for(int pc=0; pc < mc; pc += MR){   //2-nd loop
                    int mr = min(MR, mc-pc);
                    double *A_ptr2 = A_ptr3 + pc*K;
                    double *C_ptr2 = C_ptr3 + pc*N;
                    for(int qc=0; qc < nc; qc +=NR){ //1-st loop
                        int nr = min(NR, nc-qc);
                        double *B_ptr1 = B_ptr4 + qc*K;
						double *C_ptr1 = C_ptr2 + qc;
						//printf("qc=%d, nc=%d\n",qc,nc);
						//printf("A offset=%d, B offset=%d, C offset=%d\n",A_ptr2-A, B_ptr1-B, C_ptr1-C);
                        // rank-1 update, get a 4x4 of C
                        // matrix A mr x kc, matrix B kc x nr
						/*
                        for(int i=0; i<mr; i++)
                            for(int j=0; j<nr; j++)
                                for(int k=0; k<kc; k++)
                                C_ptr1[i*N+j] += A_ptr2[i*K + k] * B_ptr1[j*K + k];
                        */
                        if(mr == 1 && nr == 1)
                            rank_1x1(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 1 && nr == 2)
                            rank_1x2(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 1 && nr == 3)
                            rank_1x3(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 1 && nr == 4)
                            rank_1x4(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 2 && nr == 1)
                            rank_2x1(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 2 && nr == 2)
                            rank_2x2(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 2 && nr == 3)
                            rank_2x3(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 2 && nr == 4)
                            rank_2x4(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 3 && nr == 1)
                            rank_3x1(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 3 && nr == 2)
                            rank_3x2(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 3 && nr == 3)
                            rank_3x3(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 3 && nr == 4)
                            rank_3x4(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 4 && nr == 1)
                            rank_4x1(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 4 && nr == 2)
                            rank_4x2(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 4 && nr == 3)
                            rank_4x3(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                        else if(mr == 4 && nr == 4)
                            rank_4x4(A_ptr2,B_ptr1,C_ptr1,kc,M,N,K);
                    }
                }
            }
        }
    }
}

void mm(double *a, double *b, double* result, int M, int N, int K) {
  int x, y, k;
  for (y = 0; y < M; ++y) {
    for (x = 0; x < N; ++x) {
      double r = 0.0;
      for (k = 0; k < K; ++k) {
        r += a[y * K + k] *  b[x*K + k];
	  	//printf("%d = %d * %d	",(int)r,(int)a[y * K + k],(int)b[x*K + k]);
      }
      result[y*N + x] = r;
    }
  }
}

int main(int argc, char *argv[]){
	int M =  atoi(argv[1]);
	int N =  atoi(argv[2]);
	int K =  atoi(argv[3]);

	double *A = (double*) malloc(sizeof(double) * M * K);
	double *B = (double*) malloc(sizeof(double) * K * N);
	double *C = (double*) malloc(sizeof(double) * M * N);
	double *ref_C = (double*) malloc(sizeof(double) * M * N);

	/*initialize matrices*/
	/*A is row major*/
//	printf("matrix A:\n");
	for(int i=0; i<M; i++){
		for(int k=0; k < K; k++){
			A[i*K + k] = i*K + k;
//			printf("%d	",(int)A[i*K+k]);
		}
//		printf("\n");
	}

	/*B is column major*/
//	printf("matrix B:\n");
	for(int k=0; k<K; k++){
		for(int j=0; j<N; j++){
			B[j*K + k] = j*K + k;
//			printf("%d	",(int)B[j*K+k]);
		}
//		printf("\n");
	}

	/*C is row major*/
	for(int i=0; i<M; i++)
		for(int j=0; j<N; j++)
			C[i*N+j] = 0;

	gemm(A,B,C,M,N,K);
/*	
	printf("My result:\n");
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++)
			printf("%d	",(int)C[i*N+j]);
		printf("\n");
	}
*/

	/*C is row major*/
	for(int i=0; i<M; i++)
		for(int j=0; j<N; j++)
			ref_C[i*N+j] = 0;
	mm(A,B,ref_C,M,N,K);

//	printf("Reference result:\n");
/*
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			if(C[i*N+j] != ref_C[i*N+j])
			printf("different: %d 	",i*N+j);
		}
	}
*/	

}
