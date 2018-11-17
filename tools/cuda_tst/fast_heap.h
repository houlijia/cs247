#ifndef __fast_heap_h__
#define __fast_heap_h__

/** \file
   Declarations for heaps based on ObjectMultiPool
 */ 

#include "ObjectMultiPool.h"
#include "HeapElement.h"
#include "GlobalPtr.h"

//* FastHeap is a ObjectMultiPool with a default constructor
template <typename spec_t>
class FastHeap
  : public ObjectMultiPool<HeapElement<spec_t>, spec_t>
{
public:
  FastHeap() :
    ObjectMultiPool<HeapElement<spec_t>, spec_t>(spec_t(), 256) {}
    
  virtual ~FastHeap() {}
    
    
};

typedef FastHeap<HeapElementSpec> fast_heap_t;

extern GlobalPtr<fast_heap_t> fast_heap;

#ifdef __NVCC__

typedef FastHeap<D_HeapElementSpec> d_fast_heap_t;
typedef FastHeap<H_HeapElementSpec> h_fast_heap_t;

extern GlobalPtr<d_fast_heap_t> d_fast_heap;
extern GlobalPtr<h_fast_heap_t> h_fast_heap;

#endif	// __NVCC__

//* Asserts that no elements are out clear all heaps.
void reset_fast_heap();

#endif	// __fast_heap_h__
