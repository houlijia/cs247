#ifndef COMMON_LJW_H
#define COMMON_LJW_H

#include <limits.h>
#include <error.h>

#include <stdlib.h>
#include <string.h>

//switch for the asserts
#define DEBUG

//place to store global error string, used by the typedef below
static char* globalError = NULL;


//for cross-platform types 
#if ULONG_MAX == 0xFFFFFFFFFFFFFFFF 
//globalError = "uing 1";
typedef unsigned long longInt;
#elif ULLONG_MAX == 0xFFFFFFFFFFFFFFFF  
typedef unsigned long long longInt;
//globalError = "uing 12";
#else
globalError = "no long int type supported on this machines";
#endif

//for cross-platform types 
#if LONG_MAX == 0x7FFFFFFFFFFFFFFF 
typedef long SlongInt;
#elif LLONG_MAX == 0x7FFFFFFFFFFFFFFF  
typedef long long SlongInt;
#else
globalError = "no signed long int type supported on this machines";
#endif


//for cross-platform types 
#if USHRT_MAX== 0xFFFFFFFF
typedef unsigned short uint32;
#elif UINT_MAX ==  0xFFFFFFFF 
typedef unsigned int uint32;
#elif ULONG_MAX ==  0xFFFFFFFF 
typedef unsigned long uint32;
#else
globalError = "no uint32 type supported on this machines";
#endif

//for cross-platform types 
#if USHRT_MAX== 0xFFFF
typedef unsigned short uint16;
#elif UINT_MAX ==  0xFFFF
typedef unsigned int uint16;
#else
globalError = "no uint32 type supported on this machines";
#endif


//for cross-platform types 
#if UCHAR_MAX== 0xFF
typedef unsigned char uint8;
#else
globalError = "no uint32 type supported on this machines";
#endif


//some general defines
#define TRUE 1
#define FALSE 0

//inline longInt MAX(longInt a, longInt b);

//inline longInt MAX_TR(longInt a, longInt b, longInt c);

#ifdef __cplusplus
extern "C"
{
#endif

extern void checkError (const char* errString);





#ifdef __cplusplus
}
#endif



#endif
