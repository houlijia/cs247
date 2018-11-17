#ifndef __HeapElement_h__
#define __HeapElement_h__

/** \file
    This file contains class declaration for classes of objects representing raw
    memory allocated from Heap
 */ 

#include "ObjectPool.h"

#ifdef __NVCC__
#include "CudaDevInfo.h"
#endif

/** Specifier for HeapElement taken from the usual stack */
class HeapElementSpec
{
public:
  // Constructors
  HeapElementSpec(size_t sz_ = 0)
    : sz(sz_) {}
  HeapElementSpec(const HeapElementSpec &other)
    : sz(other.size()) {}

  size_t size() const { return sz; }
  void setSize(size_t val) { sz = val; }

  bool onGpu() const { return false; }

  //* Allocate a vector of size_t bytes.
  //* \return pointer to the allocated vector
  static void *allocate(size_t nb, //*< number of bytes to allocate
			bool *on_g //*< If not NULL, returns true if
			//*allocated vector is on GPU
			) {
    if (on_g != NULL)
      *on_g = false;

    return new char[nb];
  }

  //* Deallocate vector allocated by allocate()
  static void deallocate(void * vec) { delete [] (char *) vec; }

private:
  size_t sz;			//*< object size
};				// HeapElementSpec


#if defined(__NVCC__)

/** Specifier for HeapElement allocated on GPU */
class D_HeapElementSpec
  : public HeapElementSpec
{
public:
  D_HeapElementSpec(size_t sz_ = 0)
    : HeapElementSpec(sz_) {}
  D_HeapElementSpec(const HeapElementSpec & other)
    : HeapElementSpec(other) {}

  bool onGpu() const { return true; }

  //* Allocate a vector of size_t bytes.
  //* \return pointer to the allocated vector
  static void *allocate(size_t nb, //*< number of bytes to allocate
			bool *on_g //*< If not NULL, returns true if
			//*allocated vector is on GPU
			) {
    if (on_g != NULL)
      *on_g = true;

    void *vec;
    gpuErrChk(cudaMalloc(&vec, nb), 
	      "cuda_ObjectMultiPool:D_HeapElementSpec:alloc_error", "");
    return vec;
  }

  //* Deallocate vector allocated by allocate()
  static void deallocate(void * vec) {
    gpuErrChk(cudaFree(vec), "cuda_ObjectMultiPool:D_HeapElementSpec:alloc_error", "");
  }
};				// D_HeapElementSpec

/** Specifier for HeapElement allocated on page-locked memory on CPU */
class H_HeapElementSpec
  : public HeapElementSpec
{
public:
  H_HeapElementSpec(size_t sz_ = 0)
    : HeapElementSpec(sz_) {}
  H_HeapElementSpec(const HeapElementSpec & other)
    : HeapElementSpec(other) {}

  bool onGpu() const { return false; }

  //* Allocate a vector of size_t bytes.
  //* \return pointer to the allocated vector
  static void *allocate(size_t nb, //*< number of bytes to allocate
			bool *on_g //*< If not NULL, returns true if
			//*allocated vector is on GPU
			) {
    if (on_g != NULL)
      *on_g = false;

    void *vec;
    gpuErrChk(cudaMallocHost(&vec, nb), 
	      "cuda_ObjectMultiPool:H_HeapElementSpec:alloc_error", "");
    return vec;
  }

  //* Deallocate vector allocated by allocate()
  static void deallocate(void * vec) {
    gpuErrChk(cudaFreeHost(vec), "cuda_ObjectMultiPool:H_HeapElementSpec:alloc_error", "");
  }
};				// H_HeapElementSpec
#endif	// defined(__NVCC__)

//* Generic base class for heap elements.
class GenericHeapElement: public GenericPoolObject
{
public:
  GenericHeapElement(GenericObjectPool * const pool_,
		     void *data_
		     )
    : GenericPoolObject(pool_), data((char *)data_) 
  {};
  
  //* Destructor. \note NOT Virtual: I do not expect to see pointers to
  //* GenericPoolObject 
  ~GenericHeapElement() {};

  //* Operator * returns the allocated data. This makes GenericHeapElement
  //* behave like a void** variable
  void * operator * () { return data; }
  const void *operator * () const {return data; }

protected:
  char *data;
};

//* Template of classes for heap elements, parameterized by the type specifier.
template<class heap_spec_t>
class HeapElement: public GenericHeapElement
{
public:
  HeapElement(const heap_spec_t &spec_, GenericObjectPool * const pool_)
    : GenericHeapElement(pool_, spec_.allocate(spec_.size()*sizeof(char), &on_gpu)),
      spec(spec_)
  {}

  ~HeapElement()
  { if(**this != NULL) spec.deallocate(**this); }

  size_t size() const {return spec.size();}
  bool onGpu() const { return on_gpu; }

private:
  const heap_spec_t &spec;
  bool on_gpu;
};

#endif	// __HeapElement_h__
