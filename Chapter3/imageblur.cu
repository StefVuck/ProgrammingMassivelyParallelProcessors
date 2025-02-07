#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

const int BLUR_SIZE = 2;

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        
        for(int blurRow=-BLUR_SIZE; blurRow<BLUR_SIZE+1; ++blurRow) {
            for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                
                if(curRow>=0 && curRow<h && curCol>=0 && curCol<w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels;
                }
            }
        }
        out[row*w + col] = (unsigned char)(pixVal/pixels);
    }
}

int main() {
    // Create test image (grayscale gradient)
    const int width = 256;
    const int height = 256;
    unsigned char* h_in = new unsigned char[width * height];
    unsigned char* h_out = new unsigned char[width * height];
    
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            h_in[i*width + j] = (i + j) % 256;  // Create gradient pattern
        }
    }

    // Allocate device memory
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, width * height * sizeof(unsigned char));
    cudaMalloc(&d_out, width * height * sizeof(unsigned char));
    
    // Copy input to device
    cudaMemcpy(d_in, h_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    blurKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    
    // Copy result back
    cudaMemcpy(h_out, d_out, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save input and output as raw files for comparison
    std::ofstream fout_in("input.raw", std::ios::binary);
    fout_in.write((char*)h_in, width * height);
    fout_in.close();

    std::ofstream fout_out("output.raw", std::ios::binary);
    fout_out.write((char*)h_out, width * height);
    fout_out.close();

    // Cleanup
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);
    
    std::cout << "Files saved: input.raw and output.raw\n";
    return 0;
}
