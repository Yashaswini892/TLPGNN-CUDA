%%cuda
#include <cuda.h>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>


template <typename scalar_t>
__global__ void gcn_conv_cuda_forward_kernel(
        const int n_vertex,
        const int fsize,
        scalar_t * __restrict__ features,
        int * __restrict__ col_starts,
        int * __restrict__ rows,
        scalar_t * __restrict__ result) {

    int des_v;
    des_v = blockIdx.x * blockDim.y + threadIdx.y;
    printf("%d ", des_v);



    if (des_v < n_vertex)
    {
        int s_pos = col_starts[des_v];
        int e_pos = col_starts[des_v+1];
        float deg;
        if (e_pos - s_pos == 0){
            deg = 1.0;
        }
        else {
           deg = 1.0 / (e_pos - s_pos);
        }

        scalar_t * des_p = result + des_v * fsize;
        for (int k = threadIdx.x; k < fsize; k += blockDim.x) {
            float ret = 0.0;
            for (int i = s_pos; i < e_pos; ++i)
            {
                ret += features[rows[i] * fsize + k];

            }
            float var = ret*deg;
            des_p[k] = var;

        }
    }
}

void gcn_conv_cuda_forward(int n_vertex, int fsize, float *features, int *col_starts, int *rows, float *result) {
    const int yaxis = 1;
    const int xaxis = fsize <= 16 ? 16 : 32;
    const dim3 threads(xaxis, yaxis);
    const int blocks = (n_vertex + yaxis - 1) / yaxis;

    gcn_conv_cuda_forward_kernel<<<blocks, threads>>>(n_vertex, fsize, features, col_starts, rows, result);
}

int main() {
    long long int num_vertices, feature_size;

    float *features, *result;
    int *col_starts, *rows;


    std::ifstream featuresFile("/content/features.txt");
    featuresFile >> num_vertices >> feature_size;
    std::cout << "Number of vertices: " << num_vertices << ", Feature size: " << feature_size << std::endl;

    float *h_features = new float[num_vertices * feature_size];
    for (int i = 0; i < num_vertices * feature_size; ++i) {
        featuresFile >> h_features[i];
    }
    cudaMalloc((void**)&features, num_vertices * feature_size * sizeof(float));
    cudaMemcpy(features, h_features, num_vertices * feature_size * sizeof(float), cudaMemcpyHostToDevice);

    std::ifstream rowsFile("/content/rows.txt");
    std::vector<int> rowsVec;
    int tempRow;
    while (rowsFile >> tempRow) {
        rowsVec.push_back(tempRow);
    }
    int *h_rows = rowsVec.data();
    cudaMalloc((void**)&rows, rowsVec.size() * sizeof(int));
    cudaMemcpy(rows, h_rows, rowsVec.size() * sizeof(int), cudaMemcpyHostToDevice);


    std::ifstream colStartsFile("/content/col_starts.txt");
    std::vector<int> colStartsVec;
    int tempColStart;
    while (colStartsFile >> tempColStart) {
        colStartsVec.push_back(tempColStart);
    }
    int *h_col_starts = colStartsVec.data();
    cudaMalloc((void**)&col_starts, colStartsVec.size() * sizeof(int));
    cudaMemcpy(col_starts, h_col_starts, colStartsVec.size() * sizeof(int), cudaMemcpyHostToDevice);

    float *h_result =  new float[num_vertices * feature_size];
    memcpy(h_result, h_features, num_vertices * feature_size * sizeof(float));
    cudaMalloc((void**)&result, num_vertices * feature_size * sizeof(float));
    cudaMemcpy(result, h_result, num_vertices * feature_size * sizeof(float), cudaMemcpyHostToDevice);

    gcn_conv_cuda_forward(num_vertices, feature_size, features, col_starts, rows, result);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, result, num_vertices * feature_size * sizeof(float), cudaMemcpyDeviceToHost);



    cudaFree(features);
    cudaFree(col_starts);
    cudaFree(rows);
    cudaFree(result);

    return 0;
}
