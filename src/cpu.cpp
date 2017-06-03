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


void pixelWarpKernel(
	const unsigned char *InImageArray,
	unsigned char *OutImageArray,
	const int FaceOffset[6][2],
	const int Face3DCoordsFactor[6][3][2],
	const int Face3DCoordsConstant[6][3],
	const int outBlockWidth,
	int face, int inFaceX, int inFaceY) {

	int inImageWidth = outBlockWidth << 2;
	int inImageHeight = outBlockWidth << 1;
	int outImageWidth = outBlockWidth << 2;
	int outImageHeight = outBlockWidth * 3;

	// 2D coords in this face within [0..2]
	float inFaceX_2 = (2.f * inFaceX) / outBlockWidth;
	float inFaceY_2 = (2.f * inFaceY) / outBlockWidth;

	// 2D coords in the whole output file
	int outX = inFaceX + FaceOffset[face][0];
	int outY = inFaceY + FaceOffset[face][1];

	// 3D coords on the cube
	float cubeX = Face3DCoordsConstant[face][0] +
		Face3DCoordsFactor[face][0][0] * inFaceX_2 +
		Face3DCoordsFactor[face][0][1] * inFaceY_2;
	float cubeY = Face3DCoordsConstant[face][1] +
		Face3DCoordsFactor[face][1][0] * inFaceX_2 +
		Face3DCoordsFactor[face][1][1] * inFaceY_2;
	float cubeZ = Face3DCoordsConstant[face][2] +
		Face3DCoordsFactor[face][2][0] * inFaceX_2 +
		Face3DCoordsFactor[face][2][1] * inFaceY_2;

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
	(InImageArray[(x * inImageWidth + y) * 3 + offset])

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
	OutImageArray[(x * outImageWidth + y) * 3 + offset]

	_setColor(outX, outY, 0) = (int)round(colorB);
	_setColor(outX, outY, 1) = (int)round(colorG);
	_setColor(outX, outY, 2) = (int)round(colorR);
}

//    [T]           |    [0]
// [L][F][R][B]     | [1][2][3][4]
//    [D]           |    [5]
// target structure | face idx
int convertWithCPU(
	int outBlockWidth,
	unsigned char *inImageArray,
	unsigned char *outImageArray) {

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

	for (int face = 0; face < 6; face++)
		for (int i = 0; i < outBlockWidth; i++)
			for (int j = 0; j < outBlockWidth; j++)
				pixelWarpKernel(
					inImageArray,
					outImageArray,
					(int (*) [2])faceOffset,
					(int (*) [3][2])face3DCoordsFactor,
					(int (*) [3])face3DCoordsConstant,
					outBlockWidth,
					face, i, j);
	return 0;
}

int mainPrecess(IplImage *inImageMat, IplImage **outImageMat) {
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

	int ret = convertWithCPU(outBlockWidth, inImageArray, outImageArray);

	if (ret != 0)
	{
		fprintf(stderr, "convertWithCPU failed!");
		return 1;
	}

	*outImageMat = cvCreateImageHeader(cvSize(outImageWidth, outImageHeight), IPL_DEPTH_8U, 3);
	cvSetData(*outImageMat, outImageArray, outImageWidth * 3);

	return 0;
}

int main(int argc, char** argv) {
	int ret;

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

	printf("cpu time %fs\n", (double)(stop_time - start_time) / CLOCKS_PER_SEC);

	ret = writeImage(outImageName, outImageMat);
	if (ret != 0) {
		fprintf(stderr, "writeImage failed!");
		return 1;
	}

	return 0;
}
