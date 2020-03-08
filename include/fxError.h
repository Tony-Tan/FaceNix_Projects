#ifndef FX_ERROR_H
#define FX_ERROR_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_FUNCTION_NAME_LENGTH 100

#define __BEGIN__   
#define __EXIT__		goto exit;
#define __END__		goto exit;exit:;
	

#define FX_ERROR_SIZE_NEGATIVE			-1
#define FX_ERROR_MALLOC_MEMROY_FAIL		-2
#define FX_ERROR_NULL_POINTER			-3
#define FX_ERROR_DATA_TYPE_WRONG		-4
#define FX_ERROR_MAT_SIZE_UNEQUAL		-5
#define FX_ERROR_MAT_DATA_NULL          -6
#define FX_ERROR_POSITION_OUTOFRANGE	-7
#define FX_ERROR_PROCESSBAR_PARAM		-8
#define FX_ERROR_PARAM_NEGATIVE_WRONG   -9

char FUNCTIONNAME[100];

void FX_FUNCTION(char * name);

#define FX_SIZE_POSITIVE_TEST(size)													\
    if((size).width<0||(size).height<0){											\
	fxError(FX_ERROR_SIZE_NEGATIVE, FUNCTIONNAME, __FILE__, __LINE__);					\
}

#define FX_MALLOC_MEMORY_TEST(memory_address)									\
if (memory_address==NULL){														\
	fxError(FX_ERROR_MALLOC_MEMROY_FAIL, FUNCTIONNAME, __FILE__, __LINE__);			\
}

#define FX_NULL_POINTER_TEST(pointer)												\
	if (pointer==NULL){																\
	fxError(FX_ERROR_NULL_POINTER, FUNCTIONNAME, __FILE__, __LINE__);					\
}

#define FX_MAT_DATA_TEST(mat)												\
if (((mat)->data)==NULL){																\
	fxError(FX_ERROR_NULL_POINTER, FUNCTIONNAME, __FILE__, __LINE__);					\
}

#define FX_DATA_TYPE_TEST(type1,type2)													\
if ((type1) != (type2)){																\
	fxError(FX_ERROR_DATA_TYPE_WRONG, FUNCTIONNAME, __FILE__, __LINE__);					\
}


#define FX_MAT_SIZE_TEST(mat_pointer1,mat_pointer2)							\
if ((mat_pointer1->width) != (mat_pointer2->width)||								\
	(mat_pointer1->height) != (mat_pointer2->height)){								\
	fxError(FX_ERROR_MAT_SIZE_UNEQUAL, FUNCTIONNAME, __FILE__, __LINE__);					\
}


#define FX_POINT_OUT_OF_RANGE_TEST(mat,point)									\
if (((mat).width) < ((point).x) ||												\
	((mat).height) < ((point).y) ||												\
	((point).x)<0 ||															\
	((point).y)<0)																\
{																					\
	fxError(FX_ERROR_POSITION_OUTOFRANGE, FUNCTIONNAME, __FILE__, __LINE__);					\
}


#define FX_PROCESS_BAR_PARAM_TEST(time_i,total)									\
if (time_i>total){																\
	fxError(FX_ERROR_PROCESSBAR_PARAM, FUNCTIONNAME, __FILE__, __LINE__);	\
}
#define FX_PARAM_NEGATIVE_TEST(n)											\
if (n < 0){																	\
	fxError(FX_ERROR_PARAM_NEGATIVE_WRONG, FUNCTIONNAME, __FILE__, __LINE__);	\
}

void fxError(int error_code, char const * fun_name,char *file_path,int line);


#endif