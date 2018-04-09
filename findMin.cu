
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <iostream>


#define dllexp __declspec(dllexport)
#define byte uint8_t
#define ushort unsigned short

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

extern "C" {
	dllexp void FindMin(uint size);

	__global__ void findMin(ushort* input, uint size, ushort* value) {
		uint tid = blockIdx.x * blockDim.x + threadIdx.x;
		uint half = size / 2;
		byte odd = half % 2;

		while (half > 0) {
			if (tid == 0 && odd == 1) {
				input[tid] = input[tid] > input[half] ? input[half] : input[tid];
				input[tid] = input[tid] > input[half * 2 - 1] ? input[half * 2 - 1] : input[tid];
				*value = input[tid];
			}
			else {
				input[tid] = input[tid] > input[half + tid] ? input[half + tid] : input[tid];
			}
			__syncthreads();

			if (half == 1) {
				half = 0;
			}
			else {
				half /= 2;
				odd = half % 2;
			}
		}
	}

	void FindMin(uint size) {
		ushort* a = new ushort[size];
		ushort minValue = 0;
		ushort* dev_a = new ushort[size];
		ushort* dev_minValue = 0;

		cudaMalloc((void**)&dev_a, sizeof(ushort) * size);
		cudaMalloc((void**)&dev_minValue, sizeof(ushort));

		srand(time(NULL));
		for (int i = 0; i < size; i++) {
			a[i] = rand() % 256;
			std::cout << a[i] << " " << std::endl;
		}

		cudaMemcpy(dev_a, a, sizeof(ushort) * size, cudaMemcpyHostToDevice);

		uint threadCount = size / 2;
		printf("threadCount: %i\n", threadCount);
		cudaError_t cudaStatus;

		findMin << <1, threadCount >> > (dev_a, size, dev_minValue);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "findMin kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaMemcpy(a, dev_a, sizeof(ushort) * size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&minValue, dev_minValue, sizeof(ushort), cudaMemcpyDeviceToHost);

		std::cout << "Min value: " << minValue << std::endl;

		Error:

		cudaFree(dev_a);
		cudaFree(dev_minValue);
		delete a;
	}
