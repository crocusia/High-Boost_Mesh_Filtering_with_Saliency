#include <math_constants.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h> 
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <cub/device/device_radix_sort.cuh>

#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_DEPTH 16
#define INSERTION_SORT_THRESHOLD 32

__device__ double3 reflectDirectionD(double3 _direction, double3 _normal) {
	double3 md = mulScalar(_direction, -1.0);
	double3 result = make_double3(0.0, 0.0, 0.0);
	double w = DotD(md, _normal) * 2;
	result = mulScalar(_normal, w);
	result = subVecD(md, result);
	return Normalized(result);
}

__global__ void computeFSaliencyD(double4* _vPos, double4* _fSaliency, double* output, int* _faces, int numF) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();

	if (tid >= numF) return;

	double vSaliency = 0.0;
	for (int i = 0; i < 3; i++) {
		int fvid = _faces[tid * 3 + i];
		vSaliency = vSaliency + _vPos[fvid].w;
	}

	_fSaliency[tid] = make_double4(0.0, 0.0, 0.0, vSaliency/3);
	output[tid] = vSaliency / 3; 
}

__global__ void computeAmpFSaliencyD(double4* _fSaliency, double* _ampFSaliency, double* _lambdaSaliency, double lambda, int numF) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();
	if (tid >= numF) return;
	double lambdaSaliency = _lambdaSaliency[0];
	double fSaliency = _fSaliency[tid].w;
	if (fSaliency > lambdaSaliency) {
		_ampFSaliency[tid] = lambda * fSaliency;
	}
	else {
		_ampFSaliency[tid] = fSaliency;
	}
}

__global__ void computeDirectionVD(double4* _vPos, double4* _vSaliency, int* _vnbVstart, int* _vnbVend, int* _vnbVertex, int numV) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();
	if (tid >= numV) return;

	int start = _vnbVstart[tid];
	int end = _vnbVend[tid];
	double3 vPos = make_double3(_vPos[tid].x, _vPos[tid].y, _vPos[tid].z);
	double3 vGradient = make_double3(0.0, 0.0, 0.0);
	double vSaliency = _vPos[tid].w;

	for (int i = start; i < end + 1; i++) {
		int nvid = _vnbVertex[i];
		double3 nvPos = make_double3(_vPos[nvid].x, _vPos[nvid].y, _vPos[nvid].z);
		double nvSaliency = _vPos[nvid].w;
		double3 diff = subVecD(nvPos, vPos);
		double sdiff = nvSaliency - vSaliency;
		diff = mulScalar(Normalized(diff), sdiff);
		vGradient = sumVecD(vGradient, diff);
	}
	vGradient = Normalized(vGradient);
	_vSaliency[tid] = make_double4(vGradient.x, vGradient.y, vGradient.z, vSaliency);
}

__global__ void computeDirectionFD(double4* _normal, double4* _vSaliency, double4* _fSaliency, int* _faces, int numF) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();
	if (tid >= numF) return;
	double fSaliency = _fSaliency[tid].w;
	double3 fnormal = make_double3(_normal[tid].x, _normal[tid].y, _normal[tid].z);
	double3 avg = make_double3(0.0, 0.0, 0.0);
	for (int i = 0; i < 3; i++) {
		int vid = _faces[tid * 3 + i];
		double3 vDirection = make_double3(_vSaliency[vid].x, _vSaliency[vid].y, _vSaliency[vid].z);
		avg = sumVecD(avg, vDirection);
	}
	avg = divScalar(avg, 3);
	double3 result = reflectDirectionD(avg, fnormal);
	_fSaliency[tid] = make_double4(result.x, result.y, result.z, fSaliency);
}

__global__ void computeSmoothwihtSaliencyD(double4* _smooth, double4* _boost, double4* _fSaliency, double* _area, double* _ampFSaliency, int* fnbStart, int* fnbEnd, int* fnbFaces, double* lambdaSaliency, int numF) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();
	if (tid >= numF) return;

	double sumArea = 0.0;
	double3 areaNorm = make_double3(0.0, 0.0, 0.0);
	int start = fnbStart[tid];
	int end = fnbEnd[tid];

	for (int i = start; i < end + 1; i++) {
		int fid = fnbFaces[i];
		double area = _area[fid];
		double3 s = make_double3(_smooth[fid].x, _smooth[fid].y, _smooth[fid].z);

		if (fid != tid) {
			sumArea += area;
			double3 norm = make_double3(s.x * area, s.y * area, s.z * area);
			areaNorm = sumVecD(areaNorm, norm);
		}
		else {
			double fSaliency = _fSaliency[fid].w;
			double3 fDirection = make_double3(_fSaliency[fid].x, _fSaliency[fid].y, _fSaliency[fid].z);
			double ampSaliency = _ampFSaliency[fid];
			if (fSaliency > lambdaSaliency[0]) {
				sumArea += area * ampSaliency;
				double weight = area * ampSaliency;
				areaNorm = sumVecD(areaNorm, mulScalar(fDirection, weight));
			}
		}
	}

	double3 result = make_double3(areaNorm.x / sumArea, areaNorm.y / sumArea, areaNorm.z / sumArea);
	result = Normalized(result);
	_boost[tid] = make_double4(result.x, result.y, result.z, 0.0);
}

__global__ void computeSmoothD(double4* _smooth, double4* _boost, double* _area, int* fnbStart, int* fnbEnd, int* fnbFaces, int numF) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();
	if (tid >= numF) return;

	double sumArea = 0.0;
	double3 areaNorm = make_double3(0.0, 0.0, 0.0);
	int start = fnbStart[tid];
	int end = fnbEnd[tid];

	for (int i = start; i < end + 1; i++) {
		int fid = fnbFaces[i];
		double area = _area[fid];
		double3 s = make_double3(_smooth[fid].x, _smooth[fid].y, _smooth[fid].z);
		
		if (fid != tid) {
			sumArea += area;
			double3 norm = make_double3(s.x * area, s.y * area, s.z * area);
			areaNorm = sumVecD(areaNorm, norm);
		}
	}

	double3 result = make_double3(areaNorm.x / sumArea, areaNorm.y / sumArea, areaNorm.z / sumArea);
	result = Normalized(result);
	_boost[tid] = make_double4(result.x, result.y, result.z, 0.0);
}

__global__
void computeBoostD(double4* _normal, double4* _smooth, double4* _boost, double threshold, int numF) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();
	if (tid >= numF) return;

	double3 normal = make_double3(_normal[tid].x * (1 + threshold), _normal[tid].y * (1 + threshold), _normal[tid].z * (1 + threshold));
	double3 smooth = make_double3(_smooth[tid].x * threshold, _smooth[tid].y * threshold, _smooth[tid].z * threshold);
	double3 boost = subVecD(normal, smooth);
	boost = Normalized(boost);
	_boost[tid] = make_double4(boost.x, boost.y, boost.z, 0.0);
}

__global__
void computeBilateralFilterD(double4* _smooth, double4* _boost, double4* _vpos, int* _faces, int* fnbStart, int* fnbEnd, int* fnbFaces, double sigma_dist, double sigma_value, int numF) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__syncthreads();
	if (tid >= numF) return;

	int start = fnbStart[tid];
	int end = fnbEnd[tid];
	int3 fvid = make_int3(_faces[tid*3], _faces[tid*3+1], _faces[tid*3+2]);
	double3 fBoost = make_double3(_boost[tid].x, _boost[tid].y, _boost[tid].z);

	double w = 0.0;
	double3 result = make_double3(0.0, 0.0, 0.0);

	double3 fv1 = make_double3(_vpos[fvid.x].x, _vpos[fvid.x].y, _vpos[fvid.x].z);
	double3 fv2 = make_double3(_vpos[fvid.y].x, _vpos[fvid.y].y, _vpos[fvid.y].z);
	double3 fv3 = make_double3(_vpos[fvid.z].x, _vpos[fvid.z].y, _vpos[fvid.z].z);
	double3 fCenter = make_double3((fv1.x + fv2.x + fv3.x) / 3, (fv1.y + fv2.y + fv3.y) / 3, (fv1.z + fv2.z + fv3.z) / 3);

	for (int i = start; i < end + 1; i++) {
		int nfid = fnbFaces[i];
		int3 nfvid = make_int3(_faces[nfid*3], _faces[nfid * 3 +1], _faces[nfid * 3+2]);
		double3 nfBoost = make_double3(_boost[nfid].x, _boost[nfid].y, _boost[nfid].z);

		double3 nfv1 = make_double3(_vpos[nfvid.x].x, _vpos[nfvid.x].y, _vpos[nfvid.x].z);
		double3 nfv2 = make_double3(_vpos[nfvid.y].x, _vpos[nfvid.y].y, _vpos[nfvid.y].z);
		double3 nfv3 = make_double3(_vpos[nfvid.z].x, _vpos[nfvid.z].y, _vpos[nfvid.z].z);
		double3 nfCenter = make_double3((nfv1.x + nfv2.x + nfv3.x) / 3, (nfv1.y + nfv2.y + nfv3.y) / 3, (nfv1.z + nfv2.z + nfv3.z) / 3);
		
		double3 dist = subVecD(fCenter, nfCenter);
		double value = GetNormD(dist);
		double g_dist = gaussiandistD(value, sigma_dist);

		double3 diff = subVecD(fBoost, nfBoost);
		diff = make_double3(abs(diff.x), abs(diff.y), abs(diff.z));
		double3 g_diff = gaussiandiffD(diff, sigma_value);

		double3 g = make_double3(g_diff.x * g_dist, g_diff.y * g_dist, g_diff.z * g_dist);
		double3 bg = make_double3(g.x * nfBoost.x, g.y * nfBoost.y, g.z * nfBoost.z);
		result = sumVecD(result, bg);

		w += GetNormD(g);
	}

	_smooth[tid] = make_double4(result.x / w, result.y / w, result.z / w, 0.0);
}

extern "C"{
	void computeSaliencyDirection(double* _vPos,
		double* _normal,
		double* _vSaliency,
		double* _fSaliency,
		double* _ampSaliency,
		int* _faces,
		int* _vnbVstart,
		int* _vnbVend,
		int* _vnbVertex,
		double alpha, 
		double* lambdaSaliancy,
		double lambda,
		int numV, 
		int numF)
	{
		int numVThreads, numVBlocks;
		computeGridSizeD(numV, 64, numVBlocks, numVThreads);
		int numFThreads, numFBlocks;
		computeGridSizeD(numF, 64, numFBlocks, numFThreads);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		//Face의 Saliency를 계산하여 _fSaliency.w에 저장함
		//정렬을 위해 _ampSaliency에도 결과가 저장됨
		computeFSaliencyD <<< numFBlocks, numFThreads >>> ((double4*)_vPos, (double4*)_fSaliency, _ampSaliency, _faces, numF);

		//sort data For Saliency lambda
		thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(_ampSaliency);
		thrust::device_vector<double> d_vec(numF);
		thrust::copy(_ampSaliency, _ampSaliency + numF, d_vec.begin());
		thrust::sort(d_vec.begin(), d_vec.end());
		int a = numF * alpha;
		cudaMemcpy(lambdaSaliancy, thrust::raw_pointer_cast(d_vec.data() + a), sizeof(double), cudaMemcpyDeviceToDevice);
		
		//Face의 Saliency를 증폭하여 _ampSaliency에 저장함
		computeAmpFSaliencyD << < numFBlocks, numFThreads >> > ((double4*)_fSaliency, _ampSaliency, lambdaSaliancy, lambda, numF);
		//Vertex에 대한 Saliency 방향 계산함
		computeDirectionVD << < numVBlocks, numVThreads >> > ((double4*)_vPos, (double4*)_vSaliency, _vnbVstart, _vnbVend, _vnbVertex, numV);
		//Vertex의 Saliency 방향의 평균으로 Face Saliency 방향 벡터를 계산함
		computeDirectionFD << < numFBlocks, numFThreads >> > ((double4*)_normal, (double4*)_vSaliency, (double4*)_fSaliency, _faces, numF);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("computeSaliencyDirection execution time: %f sec\n", milliseconds / 1000);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void computeBoostNormalwithSaliency(double* normal,
		double* smooth,
		double* boost,
		double* area,
		double* fSaliency,
		double* ampSaliency,
		int* fnbStart,
		int* fnbEnd,
		int* fnbFace,
		double *lambdaSaliancy,
		int smoothK,
		double threshold,
		int numF) 
	{
		int numThreads, numBlocks;
		computeGridSizeD(numF, 64, numBlocks, numThreads);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		//Saliency 방향 기반 Boost Normal 계산
		for (int i = 0; i < smoothK; i++) {
			computeSmoothwihtSaliencyD << < numBlocks, numThreads >> > ((double4*)smooth, (double4*)boost, (double4*)fSaliency, area, ampSaliency, fnbStart, fnbEnd, fnbFace, lambdaSaliancy, numF);
			updateNormalDataD << < numBlocks, numThreads >> > ((double4*)smooth, (double4*)boost, numF);
		}
		computeBoostD << < numBlocks, numThreads >> > ((double4*)normal, (double4*)smooth, (double4*)boost, threshold, numF);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("computeBoostNormal execution time: %f sec\n", milliseconds / 1000);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

}