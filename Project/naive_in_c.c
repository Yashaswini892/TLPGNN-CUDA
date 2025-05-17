#include <stdio.h>
#include <stdlib.h>

const unsigned short grain = 1;
// const int grain = 48;

void gcn_conv_cpu_forward(
        const int n_vertex,
        const int fsize,
        float *features,
        int *col_starts,
        int *rows,
        float *result) {

    for (int des_v = 0; des_v < n_vertex; des_v++) {
        float ret;
        int s_pos = col_starts[des_v];
        int e_pos = col_starts[des_v+1];
        float deg = 1.0 / (e_pos - s_pos);
        float *des_p = result + des_v * fsize;

        for (int k = 0; k < fsize; k++) {
            ret = 0.0;
            for (int i = s_pos; i < e_pos; ++i) {
                ret += features[rows[i] * fsize + k];
            }
            des_p[k] = ret * deg;
        }
        printf("%f", des_p[1]);
        fflush(stdout); // Flush output buffer
    }
}

int main() {
    // Example usage
    const int n_vertex = 100;
    const int fsize = 10;
    const int num_cols = 10;

    // Allocate memory on host
    float *features = (float*)malloc(n_vertex * fsize * sizeof(float));
    int *col_starts = (int*)malloc((n_vertex + 1) * sizeof(int));
    int *rows = (int*)malloc(num_cols * sizeof(int));
    float *result = (float*)malloc(n_vertex * fsize * sizeof(float));
    float *grad = (float*)malloc(n_vertex * fsize * sizeof(float));
    float *indegs = (float*)malloc(n_vertex * sizeof(float));

    // Initialize data (for demonstration purposes, you should populate these with your actual data)
    for (int i = 0; i < n_vertex * fsize; i++) {
        features[i] = 1.0; // Example value
        grad[i] = 0.5; // Example value
    }

    for (int i = 0; i < n_vertex + 1; i++) {
        col_starts[i] = i * num_cols; // Example value
    }

    for (int i = 0; i < num_cols; i++) {
        rows[i] = i % n_vertex; // Example value
    }
    printf("%d", 1);
    fflush(stdout); // Flush output buffer

    // Run forward kernel
    gcn_conv_cpu_forward(n_vertex, fsize, features, col_starts, rows, result);
     
    // Run backward kernel
    // gcn_conv_cpu_backward(n_vertex, grad, indegs, col_starts, rows, fsize, result);
    printf("%d", 3);
    fflush(stdout); // Flush output buffer

    // Free memory on host
    free(features);
    free(col_starts);
    free(rows);
    free(result);
    free(grad);
    free(indegs);

    return 0;
}
