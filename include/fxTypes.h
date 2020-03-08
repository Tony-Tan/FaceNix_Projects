#ifndef FX_TYPES_H
#define FX_TYPES_H

typedef unsigned char uchar;
typedef unsigned int  uint32;
typedef unsigned long long int uint64;

typedef enum FxMatType_ FxMatType;
typedef enum FxRandomType_ FxRandomType;
typedef struct FxSize_ FxSize;
typedef struct FxPoint_ FxPoint;

typedef struct FxPoint64_ FxPoint64;
typedef struct FxMat_ FxMat;




enum FxMatType_{
	FX_8C1 = 1,
	FX_8C3 = 3,
	FX_32C1 = 4,
	FX_64C1 = 8,
	FX_32C3 = 12,
	FX_64C3 = 24
} ;

enum FxRandomType_{
	FX_RANDOM_MEAN_0,
	FX_RANDOM_MEAN_HALF_N,
	FX_RANDOM_MEAN_NEGA_HALF_N
};

struct FxSize_{
	int width;
	int height;
};
struct FxPoint_{
	int x;
	int y;
};

struct FxPoint64_{
	double x;
	double y;
};
struct FxMat_{
	uchar *data;
	int width;
	int height;
	FxMatType type;
	int width_step;

} ;
#endif