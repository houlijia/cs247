/* Copyright Alcatel-Lucent 2007 */
/* $Id: timeval.h 16 2008-04-29 18:21:43Z rma $ */

#ifndef TIMEVAL_H_GRD
#define TIMEVAL_H_GRD

#ifdef __cplusplus
#define EXT_C extern "C"
#else
#define EXT_C
#endif

#ifdef __unix
#include <sys/time.h>
#endif

#ifdef _WIN32
#include <time.h>
#include <WinSock2.h>
#include <Ws2tcpip.h>                   /* See MSDN Article ID: 257460 */
#endif


/** \file include/timeval.h declaration of functions to manipulate \c struct \c
    timeval objects */

/* For C++ compilation: Decalre prototypes as C */
#ifndef EXT_C
#ifdef __cplusplus
#define EXT_C extern "C"
#else
#define EXT_C
#endif
#endif

#ifdef _WIN32
EXT_C int gettimeofday(struct timeval *tv, struct timezone *tz);

#define sleep(x) Sleep((x)*1000)
#endif


/* Converting between timeval and timespec */
#define TIMEVAL2TIMESPEC(tspec,tval)                                    \
  ((void)(((tspec)->tv_sec=(tval)->tv_sec), ((tspec)->tv_nsec=(tval)->tv_usec*1000)))
#define TIMESPEC2TIMEVAL(tval,tspec)                                    \
  ((void)(((tval)->tv_sec=(tspec)->tv_sec), ((tval)->tv_usec=(tspec)->tv_nsec/1000)))

/* In these function it is safe to have two operands identical */

/** sum = a+b */
EXT_C void
add_timeval(struct timeval *sum,
            const struct timeval *a,
            const struct timeval *b
            );

/** diff = a-b */
EXT_C void 
sub_timeval(struct timeval *diff,
            const struct timeval *a,
            const struct timeval *b
            );

/** res=a>>b */
EXT_C void
rshft_timeval(struct timeval *res,
              const struct timeval *a,
              unsigned shft
              );

/** Convert \c ulong \c val to timeval \c *res */
EXT_C void 
ulong2timeval(struct timeval *res,
              unsigned long val
              );

/** Convert a struct timeval to a value in usec */
#define TIMEVAL2USEC(ptv) ((unsigned long)(ptv)->tv_sec*1000000 + (unsigned long)(ptv)->tv_usec)

#ifndef timersub
#define timersub(a,b,result) sub_timeval((result),(a),(b))
#endif

#ifndef timeradd
#define timeradd(a,b,result) add_timeval((result),(a),(b))
#endif

EXT_C unsigned long
timersub_usec(const struct timeval *late,
              const struct timeval *early
              );

#define timersub_msec(late,early) ((timersub_usec((late),(early))+500)/1000)

EXT_C long
timerdiff_usec(const struct timeval *late,
               const struct timeval *early
               );

EXT_C long
timerdiff_msec(const struct timeval *late,
               const struct timeval *early
               );

EXT_C void
timeradd_usec(struct timeval *tv, /**< timeval object to add to */
              long offset	/**< in uSec */
              );



EXT_C void
timeradd_msec(struct timeval *tv, /**< timeval object to add \c offset to */
              long offset               /** in mSec */
              );

/** Convert NTP time stamp to struct timeval */
EXT_C void 
ntp2timeval(unsigned long upper, unsigned long lower, struct timeval *tv);

/** Convert struct timeval NTP time stamp */
EXT_C void
timeval2ntp(const struct timeval *tv, unsigned long *upper, unsigned long *lower);


#undef EXT_C

#endif /*  TIMEVAL_H_GRD */
