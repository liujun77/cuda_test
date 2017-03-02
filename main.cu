#include <iostream>
#include <cstdlib>

using namespace std;

__global__ void add(int *a, int *b, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<n){
        a[index] += b[index];
    }
}

__global__ void rad(int *a, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<n){
        a[index] = 1;
    }
}

int main(){
    int N = 10000;
    int M = 512;
    int *a, *b;
    int *d_a, *d_b;
    int size = N * sizeof(int);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    a = (int *)malloc(size);
    b = (int *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    rad<<<(N+M-1)/M, M>>>(d_a, size);
    rad<<<(N+M-1)/M, M>>>(d_b, size);
    add<<<(N+M-1)/M, M>>>(d_a, d_b, size);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
    
    int ret = 0;
    for(int i=0; i<N; i++)
        ret += a[i];
    cout << ret << endl;
    free(a); free(b);
    cudaFree(d_a); cudaFree(d_b);
    return 0;
}
