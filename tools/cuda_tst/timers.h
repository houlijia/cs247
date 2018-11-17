#ifndef __timers_h__
#define __timers_h__

/*
  \file
*/

#include "GlobalPtr.h"
#include <stdio.h>

#define HAS_GETTIME (defined(__linux) || defined(__linux__) || defined(linux) || defined(__GNUC__))
#if HAS_GETTIME

#include <time.h>
#include <errno.h>
#include <string.h>
#include <vector>

#ifdef MATLAB_MEX_FILE
#include "mex_tools.h"
#endif

#include "mex_assert.h"

template<clockid_t clock_id>
class TimeReaderGettime
{
public:
  TimeReaderGettime();

  //* capture first time entry and save it
  void tick()
  { int err = clock_gettime(clock_id, &start);
    (void) err;
    mex_assert(!err,
	       ("TimerReaderGettime:error",
	      "Error (%d) in clock_gettime: %s",errno, strerror(errno)));
     
  }
  
  //* set first time entry of this timer to other timer's
  //* second time entry
  void tick(const TimeReaderGettime<clock_id> &other)
  { start = other.stop; }
  
  //* capture second time entry and save it
  void tock()
  { int err = clock_gettime(clock_id, &stop);
    (void) err;
    mex_assert(!err,
	       ("TimerReaderGettime:error",
	      "Error (%d) in clock_gettime: %s",errno, strerror(errno)));
     
  }
  
  //* return the difference between first and second (tick and tock)
  unsigned long long ticksElapsed() const
  { return (unsigned long long)
      ((long long)(stop.tv_sec-start.tv_sec) * (long long)(1E9) +
       (long long)(stop.tv_nsec-start.tv_nsec));
  }

  //* return interval between clock ticks
  double clockInterval() const
  { return intvl; };

  //* return type of clock
  static const char *type();

private:
  struct timespec start;
  struct timespec stop;
  double intvl;  		//* interval between clock ticks
};

template<clockid_t clock_id>
TimeReaderGettime<clock_id>::TimeReaderGettime()
{
  start.tv_sec = 0; start.tv_nsec = 0;
  int err = clock_getres(clock_id, &stop);
  (void) err;			// for the case that NDBUG is defined
  mex_assert(!err,
	     ("TimerReaderGettime:error",
	      "Error (%d) in clock_getres: %s",errno, strerror(errno)));
  intvl = double(ticksElapsed()) * 1E-9;
  stop = start;  
}

typedef TimeReaderGettime<CLOCK_REALTIME> TimeReaderWC;
typedef TimeReaderGettime<CLOCK_PROCESS_CPUTIME_ID> TimeReaderCPU;
typedef TimeReaderGettime<CLOCK_THREAD_CPUTIME_ID> TimeReaderThread;

#else  // ! HAS_GETTIME

#include <math.h>
#include <omp.h>

//* Wall clock time reader using OMP timers.
class TimeReaderOmp 
{
public:
  TimeReaderOmp()
    : start(0), stop(0), intvl( 1./omp_get_wtick()) {}

  //* capture first time entry and save it
  void tick()
  { start = omp_get_wtime(); }

  //* set first time entry of this timer to other timer's
  //* second time entry
  void tick(const TimeReaderOmp &other)
  { start = other.stop; }
  
  //* capture second time entry and save it
  void tock()
  { stop = omp_get_wtime(); }
  
  //* return the difference between first and second (tick and tock)
  unsigned long long ticksElapsed() const
  { return (unsigned long long) floor(stop-start+0.5); }

  //* return interval between clock ticks
  double clockInterval() const
  { return intvl; };

  //* return type of clock
  static const char *type()
  { return "WALL"; }

private:
  double start;
  double stop;
  double intvl;  		//*< Interval in seconds
};

typedef TimeReaderOmp TimeReaderWC;

#endif				// ! HAS_GETTIME

class Timer
{

public:
  Timer(const char *name_ = "")
    : name_str(name_), on(false), cnt(0), wc_dur(0)
#if HAS_GETTIME
    , cpu_dur(0), thrd_dur(0)
#endif
  {}

  ~Timer() {}

  bool isOn() const  //*< truen if timer is running
  { return on;}

  const char *name() const	//*< returns name of timer
  { return name_str; }

  //* Start timing
  void start();

  //* start timer where other timer stopped
  void startAtStop(const Timer &other);

  //* Stop timing
  void stop();

  //* number of timer was stopped
  double count() const
  { return double(cnt); }

#if HAS_GETTIME
  enum { nReaders = 3 };
#else
  enum { nReaders = 1 };
#endif

  //* total time for k-th reader (sec.)
  double total(int k) const;

  //* type of k-th reader
  const char *type(int k) const;

  //* mean duration for k-th reader (sec.)
  double mean(int k) const;

  //* Print status of timer
  void print() const;

protected: 
  friend class Timers;

  void setName(const char *name_)
  { name_str = name_; }

private:
  // Prevent copy constructor and assignment
  Timer(const Timer &other);
  const Timer & operator= (const Timer & other);

  const char *name_str;
  bool on;
  unsigned long long cnt;

  TimeReaderWC wc;
  unsigned long long wc_dur;

#if HAS_GETTIME
  TimeReaderCPU cpu;
  unsigned long long cpu_dur;

  TimeReaderThread thrd;
  unsigned long long thrd_dur;

#endif
};

class Timers {
public:
  enum {
    TIMER_QUANT,
    TIMER_MEAN_STDV,
    N_TIMERS
  };

  Timers()
    : timers(new Timer[N_TIMERS])
  {
    for(int k=0; k<N_TIMERS; k++)
      timers[k].setName(timer_names[k]);
  }

  ~Timers()
  {
    for(int k=0; k<N_TIMERS; k++)
      timers[k].print();
    delete [] timers;
  }

  void start(int k)
  { timers[k].start(); }

  void startAtStop(int k_start, int k_stop)
  { timers[k_start].startAtStop(timers[k_stop]); }

  void stop(int  k)
  { timers[k].stop(); }

private:
  Timer *timers;

  //* An array of size N_TIMERS 
  static const char *timer_names[];
};

#ifndef USE_TIMERS
#define USE_TIMERS 0
#endif

#if USE_TIMERS

extern GlobalPtr<Timers> timers;

#define TIMER_START(k) timers->start(k)
#define TIMER_STOP(k)  timers->stop(k)
#define TIMER_START_AT_STOP(k_start, k_stop) timers->startAtStop(k_start, k_stop)

#else

#define TIMER_START(k)
#define TIMER_STOP(k)
#define TIMER_START_AT_STOP(k_start, k_stop)
#endif

#endif	// __timers_h__
