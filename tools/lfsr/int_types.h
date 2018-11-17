/* Copyright (C) 2007 Alcatel-Lucent */

/** \file int_types.h */

#ifndef _FCC_TYPE_H_
#define _FCC_TYPE_H_

#include <limits.h>

/* Type sizes */
/**
 * @brief signed 64 bit variable type
 */
#if INT_MAX ==  0x7FFFFFFFFFFFFFFF
typedef int i_64;
#define I_64_DEFINED 1
#elif LONG_MAX == 0x7FFFFFFFFFFFFFFF
typedef long i_64;
#define I_64_DEFINED 1
#elif SHRT_MAX ==  0x7FFFFFFFFFFFFFFF
typedef short i_64;
#define I_64_DEFINED 1
#elif SCHAR_MAX ==  0x7FFFFFFFFFFFFFFF
typedef signed char i_64;
#define I_64_DEFINED 1
#elif defined(LLONG_MAX)
#if LLONG_MAX ==  0x7FFFFFFFFFFFFFFFLL
typedef long long i_64;
#define I_64_DEFINED 1
#endif
#elif defined(__LONG_LONG_MAX__)
#if __LONG_LONG_MAX__ ==  0x7FFFFFFFFFFFFFFFLL
typedef long long i_64;
#define I_64_DEFINED 1
#endif
#endif

/**
 * @brief unsigned 64 bit variable type
 */
#if UINT_MAX ==  0xFFFFFFFFFFFFFFFF
typedef unsigned int u_64;
#define U_64_DEFINED 1
#elif ULONG_MAX == 0xFFFFFFFFFFFFFFFF
typedef unsigned long u_64;
#define U_64_DEFINED 1
#elif USHRT_MAX ==  0xFFFFFFFFFFFFFFFF
typedef unsigned short u_64;
#define U_64_DEFINED 1
#elif UCHAR_MAX ==  0xFFFFFFFFFFFFFFFF
typedef unsigned char u_64;
#define U_64_DEFINED 1
#elif defined(ULLONG_MAX) 
#if ULLONG_MAX ==  0xFFFFFFFFFFFFFFFFLL
typedef unsigned long long u_64;
#define U_64_DEFINED 1
#endif
#elif defined(__LONG_LONG_MAX__)
#if __LONG_LONG_MAX__ ==  0x7FFFFFFFFFFFFFFFLL
typedef unsigned long long u_64;
#define U_64_DEFINED 1
#endif
#endif


/**
 * @brief signed 32 bit variable type
 */
#if INT_MAX ==  0x7FFFFFFF
typedef int i_32;
#elif LONG_MAX == 0x7FFFFFFF
typedef long i_32;
#elif SHRT_MAX ==  0x7FFFFFFF
typedef short i_32;
#elif SCHAR_MAX ==  0x7FFFFFFF
typedef signed char i_32;
#elif defined(LLONG_MAX) && LLONG_MAX ==  0x7FFFFFFFLL
typedef long long i_32;
#else
#error no type for i_32
#endif

/**
 * @brief unsigned 32 bit variable type
 */
#if UINT_MAX ==  0xFFFFFFFF
typedef unsigned int u_32;
#elif ULONG_MAX == 0xFFFFFFFF
typedef unsigned long u_32;
#elif USHRT_MAX ==  0xFFFFFFFF
typedef unsigned short u_32;
#elif UCHAR_MAX ==  0xFFFFFFFF
typedef unsigned char u_32;
#elif defined(ULLONG_MAX) && ULLONG_MAX ==  0xFFFFFFFFLL
typedef unsigned long long u_32;
#else
#error no type for u_32
#endif

/**
 * @brief signed 16 bit variable type
 */
#if INT_MAX ==  0x7FFF
typedef int i_16;
#elif LONG_MAX == 0x7FFF
typedef long i_16;
#elif SHRT_MAX ==  0x7FFF
typedef short i_16;
#elif SCHAR_MAX ==  0x7FFF
typedef signed char i_16;
#elif defined(LLONG_MAX) &&LLONG_MAX ==  0x7FFFLL
typedef long long i_16;
#elif defined(ULLONG_MAX) &&ULLONG_MAX ==  0xFFFFLL
typedef unsigned long long i_16;
#else
#error no type for i_16
#endif

/**
 * @brief unsigned 16 bit variable type
 */
#if UINT_MAX ==  0xFFFF
typedef unsigned int u_16;
#elif ULONG_MAX == 0xFFFF
typedef unsigned long u_16;
#elif USHRT_MAX ==  0xFFFF
typedef unsigned short u_16;
#elif UCHAR_MAX ==  0xFFFF
typedef unsigned char u_16;
#else
#error no type for u_16
#endif

/**
 * @brief signed 8 bit variable type
 */
#if INT_MAX ==  0x7F
typedef int i_8;
#elif LONG_MAX == 0x7F
typedef long i_8;
#elif SHRT_MAX ==  0x7F
typedef short i_8;
#elif SCHAR_MAX ==  0x7F
typedef signed char i_8;
#elif defined(LLONG_MAX) &&LLONG_MAX ==  0x7FLL
typedef long long i_8;
#else
#error no type for i_8
#endif

/**
 * @brief unsigned 8 bit variable type
 */
#if UINT_MAX ==  0xFF
typedef unsigned int u_8;
#elif ULONG_MAX == 0xFF
typedef unsigned long u_8;
#elif USHRT_MAX ==  0xFF
typedef unsigned short u_8;
#elif UCHAR_MAX ==  0xFF
typedef unsigned char u_8;
#elif defined(ULLONG_MAX) &&ULLONG_MAX ==  0xFFLL
typedef unsigned long long i_8;
#else
#error no type for u_8
#endif

typedef u_32 chn_number_t;

/* May alias pointers for gcc ver 4 */
#if defined(__GNUC__) && __GNUC__ >= 4
#define GCC_MAY_ALIAS __attribute__((__may_alias__))
#else
#define GCC_MAY_ALIAS
#endif

typedef u_8  GCC_MAY_ALIAS u_8_alias;
typedef u_16 GCC_MAY_ALIAS u_16_alias;
typedef u_32 GCC_MAY_ALIAS u_32_alias;
typedef i_8  GCC_MAY_ALIAS i_8_alias;
typedef i_16 GCC_MAY_ALIAS i_16_alias;
typedef i_32 GCC_MAY_ALIAS i_32_alias;
#if defined(U_64_DEFINED) && U_64_DEFINED
typedef u_64 GCC_MAY_ALIAS u_64_alias;
typedef i_64 GCC_MAY_ALIAS i_64_alias;
#endif

typedef union {
    void       *v;
    u_8_alias  *b;
    u_16_alias *s;
    u_32_alias *l;
    i_8_alias  *sb;
    i_16_alias *ss;
    i_32_alias *sl;
#if defined(U_64_DEFINED) && U_64_DEFINED
    u_64_alias *ll;
    i_64_alias *sll;
#endif
} AliasPtr;

#endif	/* _FCC_TYPE_H_ */
