#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

using namespace std;

#define INF INT_MAX

__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void sortKernel(int* a, int start, int len, int distance, bool ascending) {
    int i = threadIdx.x + blockDim.x * blockIdx.x; // Thread ID
    if (i >= start && i < start + len - distance && i + distance < start + len) {
        if ((ascending && a[i] > a[i + distance]) ||
            (!ascending && a[i] < a[i + distance])) {
            swap(&a[i], &a[i + distance]);
        }
    }
}

void bitonicSort(int* a, int n) {
    int* da;  // Device array
    size_t size = n * sizeof(int);

    // Allocate memory on GPU
    cudaMalloc((void**)&da, size);
    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);

    int nThreads = 256;                     // Threads per block
    int nBlocks = (n + nThreads - 1) / nThreads; // Number of blocks

    // Sort the numbers
    for (int len = 2; len <= n; len *= 2) {
        for (int start = 0; start < n; start += len) {
            bool ascending = ((start / len) % 2 == 0);
            for (int distance = len / 2; distance > 0; distance /= 2) {
                sortKernel<<<nBlocks, nThreads>>>(da, start, len, distance, ascending);
                cudaDeviceSynchronize();  // Ensure all threads complete
            }
        }
    }

    // Copy sorted array back to host
    cudaMemcpy(a, da, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(da);
}

int nextPowerOfTwo(int n) {
    if (n <= 0) return 1;
    return pow(2, ceil(log2(n)));
}

void printArray(const int* a, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != INF) // Skip INF values
            cout << a[i] << " ";
    }
    cout << "\n";
}

int main() {
    int size;
    cout << "Enter the number of elements to sort: ";
    cin >> size;

    if (size <= 0) {
        cerr << "Invalid size. Please enter a positive number.\n";
        return -1;
    }

    int paddedSize = nextPowerOfTwo(size); // Ensure size is a power of 2
    int* a = new int[paddedSize];

    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 1000; // Generate random numbers between 0 and 999
    }
    for (int i = size; i < paddedSize; i++) {
        a[i] = INF; // Fill with INF
    }

    cout << "Unsorted array:\n";
    printArray(a, size);

    bitonicSort(a, paddedSize);

    cout << "Sorted array:\n";
    printArray(a, size);

    delete[] a;
    return 0;
}
