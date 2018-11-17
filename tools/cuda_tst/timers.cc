/*
  \file
*/

#include "mex_tools.h"
#include "timers.h"
#include "mex_assert.h"

const char *Timers::timer_names[] = {
  "quant",
  "mean_stdv"
};

#if USE_TIMERS

GlobalPtr<Timers> timers("timers");

#endif

#if HAS_GETTIME

template <> const char * TimeReaderGettime<CLOCK_REALTIME>::type()
{ return "WALL"; }
template <> const char * TimeReaderGettime<CLOCK_PROCESS_CPUTIME_ID>::type()
{ return "CPU"; }
template <> const char * TimeReaderGettime<CLOCK_THREAD_CPUTIME_ID>::type()
{ return "THRD"; }

#endif

void Timer::start()
{
  mex_assert(!isOn(),
	     ("Timer:start", "start called for timer '%'s while timer was on", name()));

  on = true;

  wc.tick();

#if HAS_GETTIME
  cpu.tick();
  thrd.tick();
#endif


}

void Timer::startAtStop(const Timer &other)
{
  mex_assert(!isOn() && other.isOn(),
	     ("Timer:start",
	      "start('%s') called for timer '%s' while timer was on or '%s' was off",
	      other.name(), name(), other.name()));

  on = true;

  wc.tick(other.wc);

#if HAS_GETTIME
  cpu.tick(other.cpu);
  thrd.tick(other.thrd);
#endif
}

void Timer::stop()
{
  mex_assert(isOn(),
	     ("Timer:start", "stop called for timer %s while timer was off", name()));

#if HAS_GETTIME
  thrd.tock();
  cpu.tock();
#endif
  wc.tock();
  
  on = false;
  ++cnt;

  wc_dur += wc.ticksElapsed();
  
#if HAS_GETTIME
  cpu_dur += cpu.ticksElapsed();
  thrd_dur += thrd.ticksElapsed();
#endif
}

double Timer::total(int k) const
{
  mex_assert(k>=0 && k<nReaders,
	     ("Timer:total", "unexpected index %d for Timer.total()", k));

  switch (k) {
  case 0: return wc.clockInterval() * double(wc_dur);

#if HAS_GETTIME
  case 1: return cpu.clockInterval() * double(cpu_dur);
  case 2: return thrd.clockInterval() * double(thrd_dur);
#endif
  default:
    return 0;
  }
}

const char * Timer::type(int k) const
{
  mex_assert(k>=0 && k<nReaders,
	     ("Timer:type", "unexpected index %d for Timer.type()", k));

  switch (k) {
  case 0: return wc.type();

#if HAS_GETTIME
  case 1: return cpu.type();
  case 2: return thrd.type();
#endif
  default:
    return "";
  }
}

double Timer::mean(int k) const
{
  mex_assert(k>=0 && k<nReaders,
	     ("Timer:mean", "unexpected index %d for Timer.mean()", k));
  
  if(!cnt) return 0;

  switch (k) {
  case 0: return wc.clockInterval() * double(wc_dur) / double(cnt);

#if HAS_GETTIME
  case 1: return cpu.clockInterval() * double(cpu_dur) / double(cnt);
  case 2: return thrd.clockInterval() * double(thrd_dur) / double(cnt);
#endif
  default:
    return 0;
  }
}

void Timer::print() const
{
  for(int k=0; k<nReaders; k++) {
    printf("TIMER %20s (%4s): %10.3f usec = %10.3f sec / %10.3f\n",
	   name(), type(k), 1E6 * mean(k), total(k), count());
	   
  }
}
