/** \file util/timeval.c

(C) Copyright Alcatel-Lucent 2006
*/

/* Unix standard header files */
#ifdef __unix
#include <sys/time.h>
#endif

#ifdef _WIN32
#include <sys/timeb.h>
#endif

/* Application headers */
#include "timeval.h"

#ifndef timercmp
#define timercmp_gt(late,early) ((((late)->tv_sec > (early)->tv_sec)) ||\
				 (((late)->tv_sec == (early)->tv_sec) &&\
				  ((late)->tv_usec > (early)->tv_usec)))
#else
#define timercmp_gt(late,early) timercmp(late,early,>)
#endif

#define MILLION (1000000)

/* sum = a+b */
void
add_timeval(struct timeval *sum,
            const struct timeval *a,
            const struct timeval *b
            )
{
  sum->tv_sec = a->tv_sec + b->tv_sec;
  sum->tv_usec = a->tv_usec + b->tv_usec;
  if(sum->tv_usec >= MILLION) {
    sum->tv_usec -= MILLION;
    sum->tv_sec += 1;
  }
}

/* diff = a-b */
void
sub_timeval(struct timeval *diff,
            const struct timeval *a,
            const struct timeval *b
            )
{
  diff->tv_sec = a->tv_sec - b->tv_sec;
  if(a->tv_usec < b->tv_usec) {
    diff->tv_sec -= 1;
    diff->tv_usec = a->tv_usec + MILLION - b->tv_usec;
  } else
    diff->tv_usec = a->tv_usec - b->tv_usec;
}

unsigned long
timersub_usec(const struct timeval *late,
              const struct timeval *early
              )
{
  struct timeval tdiff;

  timersub(late,early,&tdiff);
  return TIMEVAL2USEC(&tdiff);
}

long
timerdiff_usec(const struct timeval *late,
               const struct timeval *early
               )
{
  if(timercmp_gt(late,early))
    return (long)timersub_usec(late,early);
  else
    return -(long)timersub_usec(early,late);
}

long
timerdiff_msec(const struct timeval *late,
               const struct timeval *early
               )
{
  if(timercmp_gt(late,early))
    return (long)timersub_msec(late,early);
  else
    return -(long)timersub_msec(early,late);
}

/* res=a>>b */
void
rshft_timeval(struct timeval *res,
              const struct timeval *a,
              unsigned shft
              )
{
  int s;

  for(*res = *a; shft>0; shft -= s) {
    s = (shft>10)? 10: shft;
    res->tv_usec += res->tv_sec & ((1<<s)-1);
    res->tv_usec >>= s;
    res->tv_sec >>= s;
  }
}

/** Convert \c ulong \c val to timeval \c *res */
void
ulong2timeval(struct timeval *res,
              unsigned long val
              )
{
  res->tv_sec = val/MILLION;
  res->tv_usec = val - res->tv_sec*MILLION;
}

#if defined(_WIN32) && !defined(__MINGW32__)

/* This Windows version of gettimeofday() is taken
   from http://www.openasthra.com/c-tidbits/gettimeofday-function-for-windows/

   The gettimeofday() function obtains the current time, expressed as seconds
   and microseconds since the Epoch, and store it in the timeval structure
   pointed to by tv. As posix says gettimeoday should return zero and should
   not reserve any value for error, this function returns zero. Here is the
   program, I?ve given definition struct timezeone and for others I didn?t
   give as all other data types definitions are available in windows include
   files itself.

   RHC:  Aftr trying this function I saw that it increments in steps of
   15625, which is too large.  It was modified so that it uses
   GetSystemTimeAsFileTime() only once in order to set a reference point and
   then it uses timeGetTime() which has 1 msec resolution
*/
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
 
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  if (NULL != tv) {
    /** If true use cpuUsecS(), else use timeGetTime() */

    /** Time since the epoch until system start (usec). A value of 0
        indicates not initialized. */
    static unsigned __int64 sys_start = 0;

    /** last value of returned by timeGetTime().  Used to detect wrap
        around.
        \note: If more than 49.1 days elapse between consecutive calls to
        timeGetTime(), this mechanism of detection of wrap around does
        not work.
    */
    static unsigned __int64 last=0;
    unsigned __int64 now, tmp;

    if(!sys_start) {
      FILETIME ft;

      /* time since system start, unless 0 */
      last = now = timeGetTime();       /* msec since system start */

      GetSystemTimeAsFileTime(&ft);
 
      sys_start |= ft.dwHighDateTime;
      sys_start <<= 32;
      sys_start |= ft.dwLowDateTime;

      /*converting file time to used since unix epoch */
      sys_start /= 10;  /*convert into microseconds*/
      sys_start -= DELTA_EPOCH_IN_MICROSECS;

      /* Subtract time elapsed since system start */
      tmp = sys_start;
      sys_start -= (now*1000);
      now = tmp;
    } else {
      now = timeGetTime();      /* msec since system start */
      if(last > now)    /* Wrap around happened */
        sys_start += 1000*0x100000000LL;
      last = now;
      now = now*1000 + sys_start;
    }

    tv->tv_sec  = (long)(now / 1000000UL);
    tv->tv_usec = (long)(now % 1000000UL);
  }

  if (NULL != tz)       {
    /** Indicates whether time zore has been set */
    static int tzflag=0;
 
    if (!tzflag) {
      _tzset();
      tzflag++;
    }
    tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime = _daylight;
  }

  return 0;
}
#endif /* def _WIN32 */

void
timeradd_usec(struct timeval *tv,
              long offset
              )
{
  long q,r;

  if(offset > 0) {
    q = offset/MILLION;
    r = offset - q*MILLION;
    tv->tv_usec += r;
    if(tv->tv_usec >= MILLION) {
      tv->tv_usec -= MILLION;
      tv->tv_sec += 1;
    }
    tv->tv_sec += q;
  } else if(offset < 0) {
    offset = -offset;
    q = offset/MILLION;
    r = offset - q*MILLION;
    if(r > tv->tv_usec) {
      tv->tv_usec = (tv->tv_usec + MILLION) - r;
      tv->tv_sec -= 1;
    }
    else
      tv->tv_usec = tv->tv_usec - r;
    tv->tv_sec -= q;
  }
}

void
timeradd_msec(struct timeval *tv,
              long offset
              )
{
  long q,r;

  if(offset > 0) {
    q = offset/1000;
    r = offset - q*1000;
    tv->tv_usec += r*1000;
    if(tv->tv_usec >= MILLION) {
      tv->tv_usec -= MILLION;
      tv->tv_sec += 1;
    }
    tv->tv_sec += q;
  } else if(offset < 0) {
    offset = -offset;
    q = offset/1000;
    r = (offset - q*1000)*1000;
    if(r > tv->tv_usec) {
      tv->tv_usec = (tv->tv_usec + MILLION) - r;
      tv->tv_sec -= 1;
    }
    else
      tv->tv_usec = tv->tv_usec - r;
    tv->tv_sec -= q;
  }
}
