#include "fxError.h"

void FX_FUNCTION(char * name){
	strcpy(FUNCTIONNAME, name);
}
void fxError(int error_code,char const * fun_name,char *file_path,int line){
    switch (error_code) {
        case FX_ERROR_SIZE_NEGATIVE:
        {
            printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nMat size is Negative!\n",file_path,fun_name,line);
            exit(0);
            break;
        }
		case FX_ERROR_MALLOC_MEMROY_FAIL:
		{
			printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nMalloc Memory fail!\n", file_path, fun_name, line);
			exit(0);
			break;
		}
		case FX_ERROR_NULL_POINTER:
		{
			printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nPointer is NULL!\n", file_path, fun_name, line);
			exit(0);
			break;
		
		}
		case FX_ERROR_DATA_TYPE_WRONG:
		{
			printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nData Type isn\'t equal!\n", file_path, fun_name, line);
			exit(0);
			break;
		
		}
		case  FX_ERROR_MAT_SIZE_UNEQUAL:
		{
			printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nMat size isn\'t equal!\n", file_path, fun_name, line);
			exit(0);
			break;
		}
        case FX_ERROR_MAT_DATA_NULL:
        {
            printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nMat is Empty No data\n", file_path, fun_name, line);
            exit(0);
            break;
        }
		case FX_ERROR_POSITION_OUTOFRANGE:
		{
			printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nPosition out of range\n", file_path, fun_name, line);
			exit(0);
			break;
		}
		case FX_ERROR_PROCESSBAR_PARAM:
		{
			printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nParam Wrong!\n", file_path, fun_name, line);
			exit(0);
			break;
		}
		case FX_ERROR_PARAM_NEGATIVE_WRONG:
		{
			printf("ERROR LOCATION:\nfile:%s\nFunction:%s\nLine:%d\nParam can\'t be negative!\n", file_path, fun_name, line);
			exit(0);
			break;
		}
        default:
            break;
    }

}