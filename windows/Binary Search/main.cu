#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "info.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <fstream>
using namespace std;

const int maxn = 1e7 + 10;
const int maxm = 1e7 + 10;

int arr[maxn];// 有序序列
int queries[maxm];// 二分所要查找的数据，后面会移到GPU的Memery中
int *garr, *g_queries, *gindex;

// 随机生成数据
void make_data(int *arr, int num, const int &up)
{
	srand(time(0));
	int mod = up / num;
	arr[0] = mod;
	for (int i = 1; i < num; ++i) arr[i] = arr[i - 1] + mod;// 这里为了简单起见 直接加一个随机数 满足递增
}


// binary search using cpu
// arr index from 0
int bs_cpu(int *arr, const int & n, const int & val)
{
	int l = -1, r = n-1;
	while (r - l > 1)
	{
		int mid = (l + r)>>1;
		if (arr[mid] >= val) r = mid;
		else l = mid;
	}
	if (arr[r] == val) return r;
	return -1;//包含两种情况 即有大于等于val的值但不相同  和  全都小于val（指向最后一个数）
}


/*空的核函数，预热GPU*/
__global__ void warmup(){}


// binary search using gpu
__global__ void bs_gpu(int *garr, int *g_queries, int *gindex, int n, int query_num) {// 主机调用，设备执行
	//每个threads取出自己要处理的query
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < query_num) {
		int val = g_queries[tid];
		int l = -1, r = n - 1;
		while (r - l > 1)
		{
			int mid = (l + r) >> 1;
			if (garr[mid] >= val) r = mid;
			else l = mid;
		}
		if (garr[r] == val) gindex[tid] = r;
	}
}


//n是序列长度,up为序列中数的值域上界，即[0,up]
//out用于输出到文件 print_flag 信息是否打印到终端
//check_flag 是否对gpu的运算结果做检查（是否正确）
void solve(int n, int up, int query_num, ofstream & out, bool print_flag=false, bool check_flag=false)
{
	srand(0);
	for (int i = 0; i < query_num; ++i) {
		queries[i] = rand()*rand() % up;// record queries
	}
	if (print_flag) printf("Data Pre Processing Completed qeury_num: %d \n", query_num);

	// 计时
	float cpu_time, gpu_init_time, gpu_kernal_time;
	clock_t cpu_start, cpu_finish;
	cudaEvent_t gpu_start, gpu_finish;


	/*using cpu (serial)*/
	if(print_flag) printf("CPU processing...\n");

	//记录当前时间
	cpu_start = clock();
	for (int i = 0; i < query_num; ++i) {
		int index = bs_cpu(arr, n, queries[i]);
		//assert(index == -1 || arr[index]==queries[i]);// Test the correctness of the answer
		//printf("query: %d index: %d\n", queries[i], index);
	}
	cpu_finish = clock();
	cpu_time = (cpu_finish - cpu_start) / (float)CLOCKS_PER_SEC;
	cpu_time *= 1000;
	if(print_flag) printf("%d query_num binary search in %d length array cost %f ms with CPU\n\n", query_num, n, cpu_time);
	

	/*using gpu*/
	if (print_flag) printf("GPU processing...\n");

	//step1. init gpu(transfer data and init data)
	//记录当前时间
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_finish);
	cudaEventRecord(gpu_start, 0);
	cudaMalloc((void**)&garr, n * sizeof(int));
	cudaMalloc((void**)&g_queries, query_num * sizeof(int));
	cudaMalloc((void**)&gindex, query_num * sizeof(int));
	//copy data from host to device   
	cudaMemcpy(garr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_queries, queries, query_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(gindex, -1, query_num * sizeof(int));// init with -1
	cudaEventRecord(gpu_finish, 0);
	cudaEventSynchronize(gpu_finish);// 等待事件完成
	cudaEventElapsedTime(&gpu_init_time, gpu_start, gpu_finish);// 计算时间差


	//step2. kernel function
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_finish);
	cudaEventRecord(gpu_start, 0);
	int thread_num = 64;
	int block_num = (query_num + thread_num - 1) / thread_num;
	bs_gpu << <block_num, thread_num >> > (garr, g_queries, gindex, n, query_num);
	cudaEventRecord(gpu_finish, 0);
	cudaEventSynchronize(gpu_finish);// 等待事件完成
	cudaEventElapsedTime(&gpu_kernal_time, gpu_start, gpu_finish);// 计算时间差
	if (print_flag) printf("%d query_num binary search in %d length array cost %f (%f + %f) ms with GPU(For queries, they are parallel)\n", query_num, n, gpu_init_time+gpu_kernal_time, gpu_init_time, gpu_kernal_time);

	if (check_flag) {
		//Test the correctness of the answer
		int *index = NULL;
		index = (int *)malloc(query_num * sizeof(int));
		assert(index != NULL);
		cudaMemcpy(index, gindex, query_num * sizeof(int), cudaMemcpyDeviceToHost);//copy data from device to host  
		for (int i = 0; i < query_num; ++i) {
			assert(index[i] == -1 || arr[index[i]] == queries[i]);
			//printf("query: %d index: %d\n", queries[i], index[i]);
		}
		free(index);
	}

	//free the space  
	cudaFree(garr);
	cudaFree(g_queries);
	cudaFree(gindex);
	
	//write to file
	out<<query_num<<" "<< cpu_time <<" "<<gpu_init_time <<" "<<gpu_kernal_time<<endl;
	if (print_flag) printf("\n\n");
}

int main()
{
	//print_gpu_info();
	warmup <<<1, 1>>> ();// 预热GPU
	int n = 1e7, up = 2e8;
	assert(n <= maxn);
	make_data(arr, n, up);
	//print
	//for (int i = 0; i < n; ++i) printf("%d ", arr[i]); printf("\n");
	std::ofstream out;
	out.open("out.txt", std::ios::trunc | std::ios::out);

	int gap = 1e5;
	int sample_num = 10;
	for (int i = gap; i <= sample_num * gap; i += gap) {
		//询问
		assert(i <= maxm);
		solve(n, up, i, out, true, true);// 会将每一次的处理时间写入文件中
		//if (i % 10000 == 0) printf("now_num: %d\n", i);
	}
	
	cudaDeviceReset();//Reset
    return 0;
}