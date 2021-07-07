#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"

#define DO_PROG false
#define DO_MAXQ false
#define DO_MAXP false
#define DO_MEMD false

#define PROG_AMT 100

// Depends on input data, priority queue implementation, and adjacency list orders.
#if not DO_MAXQ
#define MAXQ 19430
#endif

#if DO_MAXQ
#define THCT 512u
#else
#define THCT 2560u
#endif
#define TH_PER_BL 64u
#define BLCT (THCT / TH_PER_BL)

#define MAX_STCT 22051
#define MAX_CRSCT 6072
#define MAX_VCT 28123
#define MAX_ECT 118314

#define FP_EPSILON 

#ifdef __INTELLISENSE__
dim3 gridDim;
dim3 blockDim;
uint3 blockIdx;
uint3 threadIdx;
int warpSize;
#endif

typedef float Ftype;

typedef unsigned int* List;

typedef Ftype* FList;

typedef unsigned int* Stack;

// A (min-) binary heap is used, as this program works with sparse graphs.
typedef unsigned int* Queue;

__device__ inline void enqueue(unsigned int& size, Queue q, unsigned int x, FList d)
{
	unsigned int c = size++, P;
	assert(size <= MAX_ECT);
	//Ftype xd = d[x.idx];
	Ftype xd = d[x];
	//while (c > 0 && x.dist < q[p = (c - 1) / 2].dist)
	while (c > 0 && xd < d[q[P = (c - 1)]])
	{
		q[c] = q[P];
		c = P;
	}
	q[c] = x;
}

__device__ inline void dequeue(unsigned int& size, Queue Q, FList D)
{
	unsigned int P = 0, c, l, r, ql, qr, qc;
	Ftype dl, dr;
	assert(size > 0);
	unsigned int x = Q[--size];
	Ftype xd = D[x];
	while ((l = P * 2 + 1) < size && xd > D[Q[c = (r = l + 1) < size && D[Q[l]] > D[Q[r]] ? r : l]])
	//while ((l = p * 2 + 1) < size && xd > (c = (r = l + 1) < size && (dl = D[ql = Q[l]]) > (dr = D[qr = Q[r]]) ? (qc = qr, dr) : (qc = ql, dl)))
	{
		Q[P] = Q[c];
		P = c;
	}
	Q[P] = x;
}

// Implementation using Brandes' algorithm
__global__ void kernelBetcA(
	unsigned int stct,
	unsigned int crsct,
	unsigned int ect,
	List stdeg,
	List crsdeg,
	List* stadj,
	List* crsadj,
	FList weights,
	FList betcaOut,
	Stack* stacks,
	Queue* queues,
	List* prevs,
	List* prevsp,
	List** prev,
	FList* dist,
	FList* sigma,
	FList* delta
#if DO_MAXQ
	, List maxQ
#endif
#if DO_MAXP
	, List maxP
#endif
#if DO_PROG
	, unsigned int* prog
#endif

)
{
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int stride = blockDim.x * gridDim.x;

	const unsigned int vct = stct + crsct;

	List* P = prev[index];
	List Ps = prevs[index];
	Stack S = stacks[index];
	Queue Q = queues[index];
	FList D = dist[index];
	FList Sp = sigma[index];
	FList Dep = delta[index];

	{
		List Psp = prevsp[index];
		unsigned int pos = 0;
		for (unsigned int stid = 0; stid < stct; stid++)
		{
			//if (stid == 0)printf("%d %d %d %p %d\n", index, stride, vct, (void*)P, stdeg[index]);
			P[stid] = Psp + pos;
			pos += stdeg[stid];
		}
		for (unsigned int crsid = 0; crsid < crsct; crsid++)
		{
			P[crsid + stct] = Psp + pos;
			pos += crsdeg[crsid];
		}
		assert(pos == 2 * ect);
	}

	unsigned int Ss = 0;
	unsigned int Qs = 0;

	for (unsigned int s = index; s < vct; s += stride)
	{
#if DO_PROG
		unsigned int pr = atomicAdd(prog, PROG_AMT);
		if (pr == 0 || pr / vct > (pr - PROG_AMT) / vct) printf("%3d/%d\n", pr / vct, PROG_AMT);
#endif
#if DO_MAXP
		unsigned int totalP = 0;
#endif

		assert(Ss == 0);
		assert(Qs == 0);
		for (unsigned int t = 0; t < vct; t++)
		{
			Ps[t] = 0;
			Sp[t] = 0.;
			D[t] = -1.;
			Dep[t] = 0.;
		}
		Sp[s] = 1.;
		D[s] = 0.;
		enqueue(Qs, Q, s, D);
#if DO_MAXQ
		if (Qs > maxQ[index]) maxQ[index] = Qs;
#endif
		//printf("Reached %d\n", index);
		while (Qs > 0)
		{
			//unsigned int v = Q->idx;
			//Ftype dv = Q->dist;
			unsigned int v = *Q;
			Ftype dv = D[v];
			dequeue(Qs, Q, D);
			const bool vst = v < stct;
			S[Ss++] = v;
			assert(Ss <= MAX_VCT);
			const unsigned int vdeg = vst ? stdeg[v] : crsdeg[v - stct];
			List vadj = vst ? stadj[v] : crsadj[v - stct];
			Ftype dw;
			if (!vst) dw = dv + weights[v - stct];
			for (unsigned int i = 0; i < vdeg; i++)
			{
				unsigned int w = vadj[i];
				if (vst)
				{
					dw = dv + weights[w];
					w += stct;
				}
				if (D[w] < 0)
				{
					D[w] = dw;
					enqueue(Qs, Q, w, D);
#if DO_MAXQ
					if (Qs > maxQ[index]) maxQ[index] = Qs;
#endif
				}
				assert(D[w] <= dw);
				if (D[w] == dw)
				{
					Sp[w] += Sp[v];
					P[w][Ps[w]++] = v;
#if DO_MAXP
					totalP++;
#endif
				}
			}
		}
#if DO_MAXP
		if (totalP > maxP[index]) maxP[index] = totalP;
#endif
		while (Ss > 0)
		{
			unsigned int w = S[--Ss];
			for (unsigned int i = 0; i < Ps[w]; i++)
			{
				unsigned int v = P[w][i];
				Dep[v] += Sp[v] / Sp[w] * (1 + Dep[w]);
			}
			if (w != s) atomicAdd(betcaOut + w, 0.5 * Dep[w]);
		}
	}
}

__host__ void compBetcA(
	unsigned int stct,
	unsigned int crsct,
	unsigned int ect,
	const unsigned int* stdeg,
	const unsigned int* crsdeg,
	const unsigned int* const* stadj,
	const unsigned int* const* crsadj,
	const Ftype* weights,
	Ftype* betcaOut
)
{
	//cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

	unsigned int vct = stct + crsct;

	cudaError_t err;

	List stdeg_d;
	err = cudaMalloc(&stdeg_d, stct * sizeof(unsigned int));
	assert(!err);
	err = cudaMemcpy(stdeg_d, stdeg, stct * sizeof(unsigned int), cudaMemcpyHostToDevice);
	assert(!err);

	List crsdeg_d;
	err = cudaMalloc(&crsdeg_d, crsct * sizeof(unsigned int));
	assert(!err);
	err = cudaMemcpy(crsdeg_d, crsdeg, crsct * sizeof(unsigned int), cudaMemcpyHostToDevice);
	assert(!err);

	List* stadj_d0;
	err = cudaMalloc(&stadj_d0, stct * sizeof(List));
	assert(!err);
	List* stadj_d1 = new List[stct];
	for (unsigned int stid = 0; stid < stct; stid++)
	{
		err = cudaMalloc(stadj_d1 + stid, stdeg[stid] * sizeof(unsigned int));
		assert(!err);
		err = cudaMemcpy(stadj_d1[stid], stadj[stid], stdeg[stid] * sizeof(unsigned int), cudaMemcpyHostToDevice);
		assert(!err);
	}
	err = cudaMemcpy(stadj_d0, stadj_d1, stct * sizeof(List), cudaMemcpyHostToDevice);
	assert(!err);

	List* crsadj_d0;
	err = cudaMalloc(&crsadj_d0, crsct * sizeof(List));
	assert(!err);
	List* crsadj_d1 = new List[crsct];
	for (unsigned int crsid = 0; crsid < crsct; crsid++)
	{
		err = cudaMalloc(crsadj_d1 + crsid, crsdeg[crsid] * sizeof(unsigned int));
		assert(!err);
		err = cudaMemcpy(crsadj_d1[crsid], crsadj[crsid], crsdeg[crsid] * sizeof(unsigned int), cudaMemcpyHostToDevice);
		assert(!err);
	}
	err = cudaMemcpy(crsadj_d0, crsadj_d1, crsct * sizeof(List), cudaMemcpyHostToDevice);
	assert(!err);

	FList weights_d;
	err = cudaMalloc(&weights_d, crsct * sizeof(Ftype));
	assert(!err);
	err = cudaMemcpy(weights_d, weights, crsct * sizeof(Ftype), cudaMemcpyHostToDevice);
	assert(!err);

	FList betca_d;
	err = cudaMalloc(&betca_d, vct * sizeof(Ftype));
	assert(!err);
	err = cudaMemset(betca_d, 0, vct * sizeof(Ftype));
	assert(!err);

	Stack* stack_d0;
	err = cudaMalloc(&stack_d0, THCT * sizeof(Stack));
	assert(!err);
	Stack* stack_d1 = new Stack[THCT];
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		cudaMalloc(stack_d1 + thid, vct * sizeof(unsigned int));
	}
	err = cudaMemcpy(stack_d0, stack_d1, THCT * sizeof(Stack), cudaMemcpyHostToDevice);
	assert(!err);

	Queue* queue_d0;
	err = cudaMalloc(&queue_d0, THCT * sizeof(Queue));
	assert(!err);
	Queue* queue_d1 = new Queue[THCT];
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
#ifdef MAXQ
		err = cudaMalloc(queue_d1 + thid, MAXQ * sizeof(unsigned int));
#else
		err = cudaMalloc(queue_d1 + thid, (ect + 1) * sizeof(unsigned int));
#endif
		assert(!err);
	}
	err = cudaMemcpy(queue_d0, queue_d1, THCT * sizeof(Queue), cudaMemcpyHostToDevice);
	assert(!err);

	List* prevs_d0;
	err = cudaMalloc(&prevs_d0, THCT * sizeof(List));
	assert(!err);
	List* prevs_d1 = new List[THCT];
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaMalloc(prevs_d1 + thid, vct * sizeof(unsigned int));
		assert(!err);
	}
	err = cudaMemcpy(prevs_d0, prevs_d1, THCT * sizeof(List), cudaMemcpyHostToDevice);
	assert(!err);

	List* prevsp_d0;
	err = cudaMalloc(&prevsp_d0, THCT * sizeof(List));
	assert(!err);
	List* prevsp_d1 = new List[THCT];
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaMalloc(prevsp_d1 + thid, ect * 2 * sizeof(unsigned int));
		assert(!err);
	}
	err = cudaMemcpy(prevsp_d0, prevsp_d1, THCT * sizeof(List), cudaMemcpyHostToDevice);
	assert(!err);

	List** prev_d0;
	err = cudaMalloc(&prev_d0, THCT * sizeof(List*));
	assert(!err);
	List** prev_d1 = new List * [THCT];
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaMalloc(prev_d1 + thid, vct * sizeof(List));
		assert(!err);
	}
	err = cudaMemcpy(prev_d0, prev_d1, THCT * sizeof(List*), cudaMemcpyHostToDevice);
	assert(!err);

	FList* dist_d0;
	err = cudaMalloc(&dist_d0, THCT * sizeof(FList));
	assert(!err);
	FList* dist_d1 = new FList[THCT];
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaMalloc(dist_d1 + thid, vct * sizeof(Ftype));
		assert(!err);
	}
	err = cudaMemcpy(dist_d0, dist_d1, THCT * sizeof(FList), cudaMemcpyHostToDevice);
	assert(!err);

	FList* sigma_d0;
	err = cudaMalloc(&sigma_d0, THCT * sizeof(FList));
	assert(!err);
	FList* sigma_d1 = new FList[THCT];
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaMalloc(sigma_d1 + thid, vct * sizeof(Ftype));
		assert(!err);
	}
	err = cudaMemcpy(sigma_d0, sigma_d1, THCT * sizeof(FList), cudaMemcpyHostToDevice);
	assert(!err);

	FList* delta_d0;
	err = cudaMalloc(&delta_d0, THCT * sizeof(FList));
	assert(!err);
	FList* delta_d1 = new FList[THCT];
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaMalloc(delta_d1 + thid, vct * sizeof(Ftype));
		assert(!err);
	}
	err = cudaMemcpy(delta_d0, delta_d1, THCT * sizeof(FList), cudaMemcpyHostToDevice);
	assert(!err);

#if DO_PROG
	unsigned int* prog_d;
	err = cudaMalloc(&prog_d, sizeof(unsigned int));
	assert(!err);
	err = cudaMemset(prog_d, 0, sizeof(unsigned int));
	assert(!err);
#endif

#if DO_MAXQ
	List maxQ_d;
	err = cudaMalloc(&maxQ_d, THCT * sizeof(unsigned int));
	assert(!err);
	err = cudaMemset(maxQ_d, 0, THCT * sizeof(unsigned int));
	assert(!err);
#endif

#if DO_MAXP
	List maxP_d;
	err = cudaMalloc(&maxP_d, THCT * sizeof(unsigned int));
	assert(!err);
	err = cudaMemset(maxP_d, 0, THCT * sizeof(unsigned int));
	assert(!err);
#endif

	err = cudaDeviceSynchronize();
	assert(!err);

#if DO_MEMD
	size_t free, total;
	err = cudaMemGetInfo(&free, &total);
	assert(!err);
	printf("%llu free\n%llu total\n", free, total);
#endif

	kernelBetcA << <BLCT, TH_PER_BL >> > (
		stct,
		crsct,
		ect,
		stdeg_d,
		crsdeg_d,
		stadj_d0,
		crsadj_d0,
		weights_d,
		betca_d,
		stack_d0,
		queue_d0,
		prevs_d0,
		prevsp_d0,
		prev_d0,
		dist_d0,
		sigma_d0,
		delta_d0
#if DO_MAXQ
		, maxQ_d
#endif
#if DO_MAXP
		, maxP_d
#endif
#if DO_PROG
		, prog_d
#endif
		);
	err = cudaPeekAtLastError();
	assert(!err);
	err = cudaDeviceSynchronize();
	assert(!err);

	err = cudaFree(stdeg_d);
	assert(!err);

	err = cudaFree(crsdeg_d);
	assert(!err);

	err = cudaFree(stadj_d0);
	assert(!err);
	for (unsigned int stid = 0; stid < stct; stid++)
	{
		err = cudaFree(stadj_d1[stid]);
		assert(!err);
	}
	delete[] stadj_d1;

	err = cudaFree(crsadj_d0);
	assert(!err);
	for (unsigned int crsid = 0; crsid < crsct; crsid++)
	{
		err = cudaFree(crsadj_d1[crsid]);
		assert(!err);
	}
	delete[] crsadj_d1;

	err = cudaFree(weights_d);
	assert(!err);

	err = cudaMemcpy(betcaOut, betca_d, vct * sizeof(Ftype), cudaMemcpyDeviceToHost);
	assert(!err);
	err = cudaFree(betca_d);
	assert(!err);

	err = cudaFree(stack_d0);
	assert(!err);
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaFree(stack_d1[thid]);
		assert(!err);
	}
	delete[] stack_d1;

	err = cudaFree(queue_d0);
	assert(!err);
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaFree(queue_d1[thid]);
		assert(!err);
	}
	delete[] queue_d1;

	err = cudaFree(prevs_d0);
	assert(!err);
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaFree(prevs_d1[thid]);
		assert(!err);
	}
	delete[] prevs_d1;

	err = cudaFree(prevsp_d0);
	assert(!err);
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaFree(prevsp_d1[thid]);
		assert(!err);
	}
	delete[] prevsp_d1;

	err = cudaFree(prev_d0);
	assert(!err);
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaFree(prev_d1[thid]);
		assert(!err);
	}
	delete[] prev_d1;

	err = cudaFree(dist_d0);
	assert(!err);
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaFree(dist_d1[thid]);
		assert(!err);
	}
	delete[] dist_d1;

	err = cudaFree(sigma_d0);
	assert(!err);
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaFree(sigma_d1[thid]);
		assert(!err);
	}
	delete[] sigma_d1;

	err = cudaFree(delta_d0);
	assert(!err);
	for (unsigned int thid = 0; thid < THCT; thid++)
	{
		err = cudaFree(delta_d1[thid]);
		assert(!err);
	}
	delete[] delta_d1;

#if DO_PROG
	err = cudaFree(prog_d);
	assert(!err);
#endif

#if DO_MAXQ
	List maxQ = new unsigned int[THCT];
	err = cudaMemcpy(maxQ, maxQ_d, THCT * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	assert(!err);
	err = cudaFree(maxQ_d);
	assert(!err);
	unsigned int maxQPt = 0;
	for (unsigned int i = 0; i < THCT; i++)
		if (maxQ[i] > maxQPt)
			maxQPt = maxQ[i];
	delete[] maxQ;
	printf("Max Queue Size: %d\n", maxQPt);
#endif

#if DO_MAXP
	List maxP = new unsigned int[THCT];
	err = cudaMemcpy(maxP, maxP_d, THCT * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	assert(!err);
	err = cudaFree(maxP_d);
	assert(!err);
	unsigned int maxPPt = 0;
	for (unsigned int thid = 0; thid < THCT; thid++)
		if (maxP[thid] > maxPPt) maxPPt = maxP[thid];
	delete[] maxP;
	printf("Max Prev Size: %d\n", maxPPt);
#endif

}