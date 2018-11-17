#ifndef __Mutex_h__
#define __Mutex_h__

/** \file
    Definition of mutexes
*/

#ifndef MULTI_THREAD

#ifndef _MSC_VER
#define MULTI_THREAD 1
#else
#define MULTI_THREAD 0
#endif	// _MSC_VER

#endif	// MULTI_THREAD

#if MULTI_THREAD

#include <pthread.h>

#endif	// MULTI_THREAD

//* Mutex class for single thread (does nothing)
class MutexNone {
public:
  void lock() {}
  int trylock() {return 0; }
  void unlock() {}
};

#if MUTEX_THREAD

//* A class defining exceptions in mutex operations
class PthreadMutexException : public std::exception {
public:
   PthreadMutexException(int errnum_)
    : errnum(errnum_) {}
  PthreadMutexException(const PthreadMutexException &other)
    : errnum(other.errnum) {}
  
  virtual const char *what() const throw() { 
    switch(errnum) {
    case EINVAL: return "phtread mutex error - EINVAL";
    case EBUSY: return "phtread mutex error - EBUSY";
    case EAGAIN: return "phtread mutex error - EAGAIN";
    case EDEADLOCK: return "phtread mutex error - EDEADLOCK";
    case EPERM: return "phtread mutex error - EPERM";
    case ENOMEM: return "phtread mutex error - ENOMEM";
    default: return "phtread mutex error - unknown";
    }
  }
  
private:
  int errnum;

  PthreadMutexException() {}
};

class MutexPthread {
public:
  MutexPthread()
  { int err = pthread_mutex_init(&mtx, NULL);
    if(err) throw PthreadMutexException(err);
  }

  ~MutexPthread()
  { int err = pthread_mutex_destroy(&mtx);  if(err) throw PthreadMutexException(err);}
  
  void lock()
  {int err = pthread_mutex_lock(&mtx); if(err) throw PthreadMutexException(err);}

  int trylock()   {
    int err = pthread_mutex_trylock(&mtx);
    if(err && err != EBUSY) throw PthreadMutexException(err);
    return err;
  }

  void unlock()
  {int err = pthread_mutex_unlock(&mtx); if(err) throw PthreadMutexException(err);}

private:
  pthread_mutex_t mtx;
};

typedef MutexPthread Mutex

#else

typedef MutexNone Mutex;

#endif	// MUTEX_THREAD

template<typename mutex_t>
class MutexLock {
public:
  MutexLock(mutex_t& mutex) :mtx(mutex) {mtx.lock(); }
  ~MutexLock() {mtx.unlock(); }

private:
  mutex_t &mtx;
};


#endif	// __Mutex_h__
