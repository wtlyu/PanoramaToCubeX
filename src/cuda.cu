#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#ifndef M_PI
#define M_PI       3.14159265358979323846  /* pi */
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923  /* pi/2 */
#endif

#define thread_per_block 512

#define _checkCudaStatus(...) {\
	if (cudaStatus != cudaSuccess)\
	{\
		fprintf(stderr, __VA_ARGS__);\
		goto Error;\
	}\
}

#define _max(x, y) ((x < y)?y:x)

#define _min(x, y) ((x < y)?x:y)

#define _clip(x, x_min, x_max) (_max(x_min, _min(x, x_max)))

using namespace cv;

IplImage* readImage(char *imageName) {
	IplImage* tergetMat = cvLoadImage(imageName, CV_LOAD_IMAGE_COLOR);

	if(tergetMat == NULL)
		fprintf(stderr, "cvLoadImage failed!");

	return tergetMat;
}

int writeImage(char *imageName, IplImage *image) {
	int ret = cvSaveImage(imageName, image);

	if(!ret)
		fprintf(stderr, "cvSaveImage failed!");

	return !ret;
}

cudaError_t initCuda() {
	cudaError_t cudaStatus;

	int num = 0;
	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceCount(&num);
	for(int i = 0;i<num;i++)
	{
		cudaGetDeviceProperties(&prop,i);
	}
	return cudaStatus;
}

__global__ void pixelWarpKernel(
	const unsigned char *cudaInImageArray,
	unsigned char *cudaOutImageArray,
	const int cudaFaceOffset[6][2],
	const int cudaFace3DCoordsFactor[6][3][2],
	const int cudaFace3DCoordsConstant[6][3],
	const int outBlockWidth) {

	int inImageWidth = outBlockWidth << 2;
	int inImageHeight = outBlockWidth << 1;
	int outImageWidth = outBlockWidth << 2;
	int outImageHeight = outBlockWidth * 3;
	// int face = blockIdx.x;
	int face = blockIdx.x;

	int id = blockIdx.y * thread_per_block + threadIdx.x;

	// 2D coords in this face
	// int inFaceX = threadIdx.x;
	// int inFaceX = blockIdx.x;
	int inFaceX = id / outBlockWidth;
	// int inFaceY = threadIdx.y;
	// int inFaceY = blockIdx.y;
	int inFaceY = id % outBlockWidth;

	// 2D coords in this face within [0..2]
	float inFaceX_2 = (2.f * inFaceX) / outBlockWidth;
	float inFaceY_2 = (2.f * inFaceY) / outBlockWidth;

	// 2D coords in the whole output file
	int outX = inFaceX + cudaFaceOffset[face][0];
	int outY = inFaceY + cudaFaceOffset[face][1];

	// 3D coords on the cube
	float cubeX = cudaFace3DCoordsConstant[face][0] +
		cudaFace3DCoordsFactor[face][0][0] * inFaceX_2 +
		cudaFace3DCoordsFactor[face][0][1] * inFaceY_2;
	float cubeY = cudaFace3DCoordsConstant[face][1] +
		cudaFace3DCoordsFactor[face][1][0] * inFaceX_2 +
		cudaFace3DCoordsFactor[face][1][1] * inFaceY_2;
	float cubeZ = cudaFace3DCoordsConstant[face][2] +
		cudaFace3DCoordsFactor[face][2][0] * inFaceX_2 +
		cudaFace3DCoordsFactor[face][2][1] * inFaceY_2;

	float theta = atan2(cubeY, cubeX);
	float r = hypot(cubeX, cubeY);
	float phi = atan2(cubeZ, r);

	float uf = (2.f * outBlockWidth * (theta + M_PI) / M_PI);
	float vf = (2.f * outBlockWidth * (M_PI_2 - phi) / M_PI);

	int ui = floor(uf);
	int vi = floor(vf);

	int ui_2 = ui + 1;
	int vi_2 = vi + 1;

	float mu = uf - ui;
	float nu = vf - vi;

#define cornerA_x (_clip(vi, 0, inImageHeight - 1))
#define cornerA_y (ui % inImageWidth)

#define cornerB_x (_clip(vi, 0, inImageHeight - 1))
#define cornerB_y (ui_2 % inImageWidth)

#define cornerC_x (_clip(vi_2, 0, inImageHeight - 1))
#define cornerC_y (ui % inImageWidth)

#define cornerD_x (_clip(vi_2, 0, inImageHeight - 1))
#define cornerD_y (ui_2 % inImageWidth)

#define _getColor(x, y, offset) \
	(cudaInImageArray[(x * inImageWidth + y) * 3 + offset])

	float colorB =
		_getColor(cornerA_x, cornerA_y, 0) * (1 - mu) * (1 - nu) +
		_getColor(cornerB_x, cornerB_y, 0) * mu * (1 - nu) +
		_getColor(cornerC_x, cornerC_y, 0) * (1 - mu) * nu +
		_getColor(cornerD_x, cornerD_y, 0) * mu * nu;

	float colorG =
		_getColor(cornerA_x, cornerA_y, 1) * (1 - mu) * (1 - nu) +
		_getColor(cornerB_x, cornerB_y, 1) * mu * (1 - nu) +
		_getColor(cornerC_x, cornerC_y, 1) * (1 - mu) * nu +
		_getColor(cornerD_x, cornerD_y, 1) * mu * nu;

	float colorR =
		_getColor(cornerA_x, cornerA_y, 2) * (1 - mu) * (1 - nu) +
		_getColor(cornerB_x, cornerB_y, 2) * mu * (1 - nu) +
		_getColor(cornerC_x, cornerC_y, 2) * (1 - mu) * nu +
		_getColor(cornerD_x, cornerD_y, 2) * mu * nu;

#define _setColor(x, y, offset) \
	cudaOutImageArray[(x * outImageWidth + y) * 3 + offset]

	_setColor(outX, outY, 0) = (int)round(colorB);
	_setColor(outX, outY, 1) = (int)round(colorG);
	_setColor(outX, outY, 2) = (int)round(colorR);
}

//    [T]           |    [0]
// [L][F][R][B]     | [1][2][3][4]
//    [D]           |    [5]
// target structure | face idx
cudaError_t convertWithCuda(
	int outBlockWidth,
	unsigned char *inImageArray,
	unsigned char *outImageArray) {

	cudaError_t cudaStatus;

	int inImageWidth = outBlockWidth << 2;
	int inImageHeight = outBlockWidth << 1;
	int outImageWidth = outBlockWidth << 2;
	int outImageHeight = outBlockWidth * 3;

	// face corner offset
	int faceOffset[6][2] = {
		{0, outBlockWidth},
		{outBlockWidth, 0},
		{outBlockWidth, outBlockWidth},
		{outBlockWidth, outBlockWidth << 1},
		{outBlockWidth, outBlockWidth * 3},
		{outBlockWidth << 1, outBlockWidth}
	};

	// 2D to 3D coords factor
	// used to convert 2D coords[0..outBlockWidth - 1] to
	// 3D coords[-1..1] on a cube
	int face3DCoordsFactor[6][3][2] = {
		{	// 0 top
			{1, 0},
			{0, 1},
			{0, 0},
		},
		{	// 1 left
			{0, 1},
			{0, 0},
			{-1, 0},
		},
		{	// 2 front
			{0, 0},
			{0, 1},
			{-1, 0},
		},
		{	// 3 right
			{0, -1},
			{0, 0},
			{-1, 0},
		},
		{	// 4 back
			{0, 0},
			{0, -1},
			{-1, 0},
		},
		{	// 5 down
			{-1, 0},
			{0, 1},
			{0, 0},
		},
	};

	// 2D to 3D coords constant
	// used to convert 2D coords[0..outBlockWidth - 1] to
	// 3D coords[-1..1] on a cube
	int face3DCoordsConstant[6][3] = {
		{-1, -1, 1},	// 0 top
		{-1, -1, 1},	// 1 left
		{1, -1, 1},	// 2 front
		{1, 1, 1},	// 3 right
		{-1, 1, 1},	// 4 back
		{1, -1, -1}	// 5 down
	};

	// Device memory pointer
	unsigned char *cudaInImageArray;
	unsigned char *cudaOutImageArray;
	int *cudaFaceOffset;
	int *cudaFace3DCoordsFactor;
	int *cudaFace3DCoordsConstant;

	// Define CUDA grid layout arrangement
	dim3 dimGrid(6, outBlockWidth * outBlockWidth / thread_per_block);
	dim3 dimBlock(thread_per_block);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	_checkCudaStatus("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	// Malloc memory space on device
	cudaStatus = cudaMalloc(
		(void**)&cudaInImageArray,
		sizeof(unsigned char) * inImageWidth * inImageHeight * 3);
	_checkCudaStatus("cudaMalloc failed!");

	cudaStatus = cudaMalloc(
		(void**)&cudaOutImageArray,
		sizeof(unsigned char) * outImageWidth * outImageHeight * 3);
	_checkCudaStatus("cudaMalloc failed!");

	cudaStatus = cudaMalloc(
		(void**)&cudaFaceOffset,
		sizeof(int) * 6 * 2);
	_checkCudaStatus("cudaMalloc failed!");

	cudaStatus = cudaMalloc(
		(void**)&cudaFace3DCoordsFactor,
		sizeof(int) * 6 * 3 * 2);
	_checkCudaStatus("cudaMalloc failed!");

	cudaStatus = cudaMalloc(
		(void**)&cudaFace3DCoordsConstant,
		sizeof(int) * 6 * 3);
	_checkCudaStatus("cudaMalloc failed!");

	// Copy data to the device
	cudaStatus = cudaMemcpy(
		cudaInImageArray,
		inImageArray,
		sizeof(unsigned char) * inImageWidth * inImageHeight * 3,
		cudaMemcpyHostToDevice);
	_checkCudaStatus("cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(
		cudaOutImageArray,
		outImageArray,
		sizeof(unsigned char) * outImageWidth * outImageHeight * 3,
		cudaMemcpyHostToDevice);
	_checkCudaStatus("cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(
		cudaFaceOffset,
		faceOffset,
		sizeof(int) * 6 * 2,
		cudaMemcpyHostToDevice);
	_checkCudaStatus("cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(
		cudaFace3DCoordsFactor,
		face3DCoordsFactor,
		sizeof(int) * 6 * 3 * 2,
		cudaMemcpyHostToDevice);
	_checkCudaStatus("cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(
		cudaFace3DCoordsConstant,
		face3DCoordsConstant,
		sizeof(int) * 6 * 3,
		cudaMemcpyHostToDevice);
	_checkCudaStatus("cudaMemcpy failed!");

	// Run kernel function
	pixelWarpKernel<<<dimGrid, dimBlock>>>(
		cudaInImageArray,
		cudaOutImageArray,
		(int (*) [2])cudaFaceOffset,
		(int (*) [3][2])cudaFace3DCoordsFactor,
		(int (*) [3])cudaFace3DCoordsConstant,
		outBlockWidth);

	// cudaThreadSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaThreadSynchronize();
	_checkCudaStatus("cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(
		outImageArray,
		cudaOutImageArray,
		sizeof(unsigned char) * outImageWidth * outImageHeight * 3,
		cudaMemcpyDeviceToHost);

	_checkCudaStatus("cudaMemcpy failed!");

Error:
	cudaFree(cudaInImageArray);
	cudaFree(cudaOutImageArray);
	cudaFree(cudaFaceOffset);
	cudaFree(cudaFace3DCoordsFactor);
	cudaFree(cudaFace3DCoordsConstant);
	return cudaStatus;
}

int mainPrecess(IplImage *inImageMat, IplImage **outImageMat) {
	cudaError_t cudaStatus;

	int inImageWidth = inImageMat -> width;
	int inImageHeight = inImageMat -> height;

	if (inImageWidth != (inImageHeight << 1)) {
		fprintf(stderr, "This image is not a standard sphare panarama image (2:1) !");
		return 1;
	}

	int outBlockWidth = inImageWidth >> 2;

	int outImageWidth = outBlockWidth * 4;
	int outImageHeight = outBlockWidth * 3;

	// Create pixel array
	unsigned char *inImageArray = (unsigned char *)(inImageMat->imageData);
	unsigned char *outImageArray = (unsigned char *)malloc(sizeof(unsigned char) * outImageWidth * outImageHeight * 3);

	// Flush output array as black color
	memset(outImageArray, 0, outImageWidth * outImageHeight);

	// Cuda Init
	cudaStatus = initCuda();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "initCuda failed!");
		return 1;
	}

	cudaStatus = convertWithCuda(outBlockWidth, inImageArray, outImageArray);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "convertWithCuda failed!");
		return 1;
	}

	*outImageMat = cvCreateImageHeader(cvSize(outImageWidth, outImageHeight), IPL_DEPTH_8U, 3);
	cvSetData(*outImageMat, outImageArray, outImageWidth * 3);

	cudaStatus = cudaThreadExit();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaThreadExit failed!");
		return 1;
	}

	return 0;
}

int main(int argc, char** argv) {
	int ret;
	cudaError_t cudaStatus;

	if(argc != 3) {
		printf("useage: %s <imagefile> <imagefile>\n ", argv[0]);
		return 1;
	}

	IplImage *inImageMat, *outImageMat;
	char* inImageName = argv[1];
	char* outImageName = argv[2];

	// Read image file and get Mat object.
	inImageMat = readImage(inImageName);
	if (inImageMat == NULL) {
		fprintf(stderr, "readImage failed!");
		return 1;
	}

	clock_t start_time = clock();

	// Main process calculation
	ret = mainPrecess(inImageMat, &outImageMat);
	if (ret != 0) {
		fprintf(stderr, "mainPrecess failed!");
		return 1;
	}

	clock_t stop_time = clock();

	printf("cuda time %fs\n", (double)(stop_time - start_time) / CLOCKS_PER_SEC);

	ret = writeImage(outImageName, outImageMat);
	if (ret != 0) {
		fprintf(stderr, "writeImage failed!");
		return 1;
	}

	return 0;
}
