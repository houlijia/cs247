#ifndef RndC_ifc_h
#define RndC_ifc_h

/* Define int32 type */

#include <stddef.h>
#include <limits.h>

struct RndCState;

#if UCHAR_MAX == 0xFFFFFFFF
typedef unsigned char RndC_uint32;
#elif USHRT_MAX == 0xFFFFFFFF
typedef unsigned short RndC_uint32;
#elif UINT_MAX == 0xFFFFFFFF
typedef unsigned int RndC_uint32;
#elif ULONG_MAX == 0xFFFFFFFF
typedef unsigned long RndC_uint32;
#else
#error no suitable type for RndC_uint32
#endif

#ifdef __cplusplus
#define EXTC extern "C"
#else
#define EXTC
#endif

/* Initialize */
EXTC void 
init_RndC(struct RndCState *state, RndC_uint32 seed);

/* Get an array of uniformly distributed random variables in [0,1] */
EXTC void 
rand_RndC(struct RndCState *rnd_state,  /* generator state, modified */
	  size_t cnt,	/* Length of output array */
	  double *out	/* output (cnt entries) */
	  );

/* Get an array of uniformly distributed random integers in  in [1,imax */
EXTC void 
randi_RndC(struct RndCState *rnd_state,  /* generator state, modified */
	   RndC_uint32 imax,	/* Maximum of random variables */
	   size_t cnt,	/* Length of output array */
	   RndC_uint32 *out	/* output (cnt entries) */
	   );

/* Get an array of normally distributed random variables (mean=0, var=1) */
EXTC void 
randn_RndC(struct RndCState *rnd_state,  /* generator state, modified */
	   size_t cnt,	/* Length of output array */
	   double *out	/* output (cnt entries) */
	   );

/* Get first cnt entries of a random permutation of 1,...,imax */
EXTC void 
randperm_RndC(struct RndCState *rnd_state,  /* generator state, modified */
	      size_t imax,
	      size_t cnt,	/* Length of output array */
	      RndC_uint32 *out	/* output (cnt entries) */
	      );

/* Get a random permutation of 1,...,imax */
EXTC void 
randperm1_RndC(struct RndCState *rnd_state,  /* generator state, modified */
	      size_t imax,
	      RndC_uint32 *out	/* output (imax entries) */
	      );
#endif	/*  RndC_ifc_h */
