#ifndef INCLUDE_FX_ALIB_H_
#define INCLUDE_FX_ALIB_H_
#define LANDMARK_TYPE 51
typedef struct FxImageData_
{
	int width;
	int height;
	unsigned char *data;
}FxImageData;

typedef enum FxCalcDevice_
{
	FXALIB_GPU = 0,
	FXALIB_CPU = 1
}FxCalcDevice;

typedef struct FxPoint_
{
	int x;
	int y;
}FxPoint;

extern class FxAlib_Imp;
class FxAlib 
{
public:
	FxAlib();
	~FxAlib();
	void cal(FxImageData *ImageData);
	void setCalcDevice(FxCalcDevice device = FXALIB_CPU);
public:
	FxImageData * image;
	FxCalcDevice CalcDevice;
	FxPoint LandMark[LANDMARK_TYPE];
private:

	FxAlib_Imp * imp;
};

#endif