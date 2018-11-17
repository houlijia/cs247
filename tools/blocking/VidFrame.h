/** \file
 * This file contains classes and functions specifying raw video frames

 * The functions in this file allow asynchronous processing (mostly copying).
 * We assume a generalized mechanism for synchronizing asynchronous
 * tasks: each asynchronous task is assigned an identifier of type SyncID
 * (several asynchronouse tasks may use the same identifier). A program can
 * wait for all task assciated with an identifier by calling a synch function
 * and supplying the identifier as an argument. This mechanism is designed to
 * model CUDA synchronization paradigm, but it may be able to model other
 * methods, such as DMA.

 */

#ifndef __VidFrame_H__
#define __VidFrame_H__

#include <stdio.h>

#include "CudaDevInfo.h"

#include "mex_assert.h"
#include "GlobalPtr.h"
#include "ObjectKeyedPool.h"

#ifndef __NVCC__
#include <string.h>

typedef int SyncID;
SyncID defaultSyncID = 0;
void blockSyncID(SyncID sync_id = defaultSyncID) { (void) sync_id; }

#else

typedef cudaStream_t SyncID;
SyncID defaultSyncID = cudaStream_t(0);
void blockSyncID(SyncID sync_id = defaultSyncID) { cudaStreamSynchronize(sync_id); }

#endif

#include "ObjectPool.h"

//* copy asynchronously \c n_elm elements of size \c elm_size from \c src to
//* \c dst. The i-th element in \c src begins at address 
//*   <tt> src + i * src_pitch </tt>, and the i-th element in dst begins at
//*  address <tt> dst + i* dst_pitch </tt>. The two arrays, \c dst and \c src,
//* are assumed to be non-overlapping.
void 
copyPitchAsync(void* dst,	//*< destination base address
	       size_t dst_pitch, //*< source pitch (bytes)
	       const void *src, //*< source base address
	       size_t src_pitch, //*< destination pitch (bytes)
	       size_t elm_size,  //*< element size (bytes)
	       size_t n_elm,	  //*< number of elements (bytes)
	       bool dst_on_gpu = false,
	       bool src_on_gpu = false,
	       SyncID sync_id = defaultSyncID
	       )
{
#ifdef __NVCC__
  cudaMemcpyKind kind = dst_on_gpu?
    (src_on_gpu? cudaMemcpyDeviceToDevice: cudaMemcpyHostToDevice):
    (src_on_gpu? cudaMemcpyDeviceToHost: cudaMemcpyHostToHost);

#if 0
  printf("%s:%d kind: %s src=0x%08lX dst=0x%08lX %lu %lu %lu %lu\n",
	    __FILE__, __LINE__,
	    kind== cudaMemcpyDeviceToDevice? "D2D": kind== cudaMemcpyHostToDevice? "H2D":
	    kind== cudaMemcpyDeviceToHost? "D2H": "H2H",
	    (ulong) src, (ulong) dst, ulong(elm_size), ulong(n_elm), ulong(src_pitch), ulong(dst_pitch));
#endif
  gpuErrChk(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, 
			      elm_size, n_elm, kind, sync_id),
	    "VidFrame:cudaMemcpy2DAsync:memcpy_error", "");
#else
  (void) sync_id;		// no synchronizatin
  (void) dst_on_gpu; (void) src_on_gpu; // no GPU

  if(src_pitch==elm_size && dst_pitch == elm_size) {
    memcpy(dst, src, n_elm*elm_size);
  }
  else {
    char *cdst = (char *) dst;
    const char *csrc = (const char *)src;
    for(size_t k=0; k<n_elm; k++) {
      memcpy(cdst, csrc, elm_size);
      csrc += src_pitch; cdst += dst_pitch;
    }
  }
#endif
}

//* Empty class for allocation and deallocation of rectangles of pixels of
//* type PixelType
template <typename PixelType>
class RectAlloc
{
public:
  static PixelType *allocate(size_t dim_v_,    //*< number of rows in the array
			   size_t vid_h_,    //*< number of columns in the array
			   size_t *pitch //*< returns the pitch value (bytes)
			   ) {
    *pitch = dim_v_ * sizeof(PixelType);
    return new PixelType[dim_v_ * vid_h_];
  }

  static void deallocate(PixelType * pxls)
  { delete [] pxls; }

  static bool onGpu() { return false; }
  static const char *name() { return "RectAlloc"; }

  typedef PixelType pixel_t;
};

#ifdef __NVCC__

//* Empty class for allocation and deallocation of rectangles of pixels of
//* type PixelType on "pinned" (page-locked) memory.
template <typename PixelType>
class H_RectAlloc
{
public:
  static PixelType *allocate(size_t dim_v_,    //*< number of rows in the array
			   size_t vid_h_,    //*< number of columns in the array
			   size_t *pitch //*< returns the pitch value
			   ) {
    *pitch = dim_v_ * sizeof(PixelType);
    PixelType* pxls;
    gpuErrChk(cudaMallocHost(&pxls, dim_v_ * vid_h_ * sizeof(PixelType)), 
	      "VidFrame:H_RectAlloc:alloc_error", "");
     return pxls;
   }

  static void deallocate(PixelType * pxls)
  {     gpuErrChk(cudaFreeHost(pxls), "VidFrame:H_RectAlloc:alloc_error", ""); }

  static bool onGpu() { return false; }
  static const char *name() { return "H_RectAlloc"; }

  typedef PixelType pixel_t;
};

//* Empty class for allocation and deallocation of rectangles of pixels of
//* type PixelType on "pinned" (page-locked) memory.
template <typename PixelType>
class D_RectAlloc
{
public:
  static PixelType *allocate(size_t dim_v_,    //*< number of rows in the array
			   size_t dim_h_,    //*< number of columns in the array
			   size_t *pitch //*< returns the pitch value
			   ) {
    PixelType* pxls;
    gpuErrChk(cudaMallocPitch(&pxls, pitch, dim_v_ * sizeof(PixelType), dim_h_), 
	      "VidFrame:D_RectAlloc:alloc_error", "");
     return pxls;
   }

  static void deallocate(PixelType * pxls)
  {     gpuErrChk(cudaFreeHost(pxls), "VidFrame:D_RectAlloc:alloc_error", ""); }

  static bool onGpu() { return true; }
  static const char *name() { return "D_RectAlloc"; }

  typedef PixelType pixel_t;
};

#endif	// #ifdef __NVCC__

template<typename alloc> class VidFrameSpec;
template<typename alloc> class ClrFrame; 

//* Specifies the dimensions of a color frame. The alloc base class is used to
//* allocae pixels.
template<typename alloc>
class ClrFrameSpec
  : private alloc
{
public:
  typedef size_t dims_t[2];

  //* Constructors, destructor and assignment
  ClrFrameSpec() {setDims(0,0); }
  ClrFrameSpec(const dims_t &dm) {setDims(dm); }
  ClrFrameSpec(size_t dim_v, size_t dim_h) { setDims(dim_v, dim_h); }
  ClrFrameSpec(const ClrFrameSpec<alloc> &other) {setDims(other.getDims()); }
  ~ClrFrameSpec() {};
  const ClrFrameSpec<alloc> & operator=(const ClrFrameSpec<alloc> &other) {
    setDims(other.getDims());
    return *this;
  }

  bool less(const ClrFrameSpec<alloc> &other) const
  { return (dimV()<other.dimV()) || (dimV()==other.dimV() && dimH()<other.dimH()); }


  size_t dimV() const { return dims[0]; }
  size_t dimH() const { return dims[1]; }
  const dims_t& getDims() const { return dims; }

  static bool onGpu() { return alloc::onGpu(); }

protected:
  class PixelArrayPtr
    : public GenericPoolObject
  {
  public:
    PixelArrayPtr(ClrFrameSpec<alloc> spec, GenericObjectPool * const pool)
      : GenericPoolObject(pool), 
	data(alloc::allocate(spec.dimV(), spec.dimH(), &ptch))
    {}
    
    ~PixelArrayPtr() { alloc::deallocate(data); }

    size_t pitch() const { return ptch; }

    typename alloc::pixel_t* operator * () { return data; }
    const typename alloc::pixel_t* operator * () const { return data; }
    
  private:
    typename alloc::pixel_t *data;
    size_t ptch;
  };

  friend class ClrFrame<alloc>;

  PixelArrayPtr *getPixelArrayPtr() const {
    PixelArrayPtr& pap = array_pool->get(*this);
    return &pap;
  }
  
private:
  dims_t dims;

  typedef ObjectKeyedPool<PixelArrayPtr, ClrFrameSpec> pool_t;
  static GlobalPtr<pool_t> array_pool;

  friend class VidFrameSpec<alloc>;
    
  void setDims(size_t dim_v, size_t dim_h)
  { dims[0] = dim_v; dims[1] = dim_h; }
  void setDims(const dims_t &dm)
  { dims[0] = dm[0]; dims[1] = dm[1]; }
};

template<typename alloc>
bool operator < (const ClrFrameSpec<alloc> &x, const ClrFrameSpec<alloc> &y)
{
  return x.less(y);
}
 
template <typename alloc>
GlobalPtr<typename ClrFrameSpec<alloc>::pool_t> 
ClrFrameSpec<alloc>::array_pool(alloc::name());

template<typename alloc> class VidFrame;

//* A color video frame class
template<typename alloc>
class ClrFrame 
  : public ClrFrameSpec<alloc> 
{
public:
  //* Constructors, destructor
  ClrFrame() : ClrFrameSpec<alloc>(), pxls_ptr(NULL), pxls(NULL), ptch(0) 
  {
#ifdef __NVCC__
    gpuErrChk(cudaStreamCreate(&strm), "ClrFrame:CudaStream_error", "");
#endif    
  }

  ClrFrame(const ClrFrameSpec<alloc>& spec) 
  {
    init(spec);
#ifdef __NVCC__
    gpuErrChk(cudaStreamCreate(&strm), "ClrFrame:CudaStream_error", "");
#endif    
  }

  ~ClrFrame() {
    if(pxls_ptr != NULL)
      pxls_ptr->discard();
#ifdef __NVCC__
    gpuErrChk(cudaStreamDestroy(strm), "ClrFrame:CudaStream_error", "");
#endif    
  }

  SyncID & syncId() const {
#ifdef __NVCC__
    return strm;
#else
    return defaultSyncID;
#endif
  }

  static bool onGpu() { return alloc::onGpu(); }
  typedef typename alloc::pixel_t pixel_t;

  size_t pitch() const
  { return ptch; }
  
  typename alloc::pixel_t * operator [] (size_t h)
  { return (pixel_t*)((char *)this->pxls + h*this->pitch()); }

  const typename alloc::pixel_t * operator [] (size_t h) const
  { return (const pixel_t *)((const char *)this->pxls + h*this->pitch()); }

  //* Copy a rectangle from the current frame into a vector asynchronously
  void copyRectangleToVectorAsync
  (const size_t bgn[2],	//*< begin indices (H,V)
   const size_t end[2],	//*< one past end indices
   typename alloc::pixel_t *data,		//*< output vector
   bool on_gpu = false,	//*< If true vector is on GPU
   SyncID sync_id = defaultSyncID
   ) const {
    copyPitchAsync
      (data,				// dst
       sizeof(typename alloc::pixel_t)*(end[0]-bgn[0]),	// dst_pitch
       (*this)[bgn[1]] + bgn[0],	// src
       this->pitch(),			// src_pitch
       sizeof(typename alloc::pixel_t)*(end[0]-bgn[0]),	// elm_size
       end[1] - bgn[1],			// n_elm
       on_gpu,				// dst_on_gpu
       this->onGpu(),			// src_on_gpu
       sync_id				// sync_id
       );
  }

  //* Copy the current frame into a vector asynchronously, using this
  //* frame's syncId()
  void copyToVectorAsync
  (typename alloc::pixel_t *data,		//*< output vector
   bool on_gpu = false	//*< If true vector is on GPU
   ) const {
    copyPitchAsync
      (data,					   // dst
       sizeof(typename alloc::pixel_t) * this->dimV(),		   // dst_pitch
       (*this)[0],				   // src
       this->pitch(),				   // src_pitch
       sizeof(typename alloc::pixel_t) * this->dimV(),		   // elm_size
       this->dimH(),				   // n_elm
       on_gpu,				   // dst_on_gpu
       this->onGpu(),				   // src_on_gpu
       syncId()				   // sync_id
       );
  }

  //* Copy a rectangle from the current frame into another ClrFrame, using the
  //* other frame's syncId()
  template<class frm_alloc_t>
  void copyRectangleToClrFrameAsync
  (const size_t bgn[2],	//*< begin indices of rectangle in this frame
   ClrFrame<frm_alloc_t> &other //*< other color frame
   ) const {
    copyPitchAsync
      (other[0],				   // dst
       other.pitch(),			   // dst_pitch
       (*this)[bgn[1]] + sizeof(typename alloc::pixel_t)*bgn[0], // src
       this->pitch(),				   // src_pitch
       sizeof(typename alloc::pixel_t)*other.dimV(),		   // elm_size
       other.dimH(),				   // n_elm
       other.onGpu(),				   // dst_on_gpu
       this->onGpu(),				   // src_on_gpu
       other.syncId()				   // sync_id
       );
  }

  //* Copy a rectangle to the current frame from a vector asynchronously
  void copyRectangleFromVectorAsync
  (const size_t bgn[2],		//*< begin indices
   const size_t end[2],		//*< one past end indices
   const typename alloc::pixel_t *data,	//*< output vector
   bool on_gpu = false	//*< If true vector is on GPU
    ) {
    copyPitchAsync
      ((*this)[bgn[1]] + sizeof(typename alloc::pixel_t)*bgn[0], // dst
       this->pitch(),				   // dst_pitch
       data,					   // src
       sizeof(typename alloc::pixel_t)*(end[0]-bgn[0]),	   // src_pitch
       sizeof(typename alloc::pixel_t)*(end[0]-bgn[0]),	   // elm_size
       end[1] - bgn[1],			   // n_elm
       this->onGpu(),				   // dst_on_gpu
       on_gpu,				   // src_on_gpu
       syncId()				   // sync_id
       );
  }

  //* Copy a vector into the current frame asynchronously
  void copyFromVectorAsync
  (const typename alloc::pixel_t *data,	//*< input vector
   bool on_gpu = false	//*< If true vector is on GPU
   ) {
    copyPitchAsync
      ((*this)[0],			  // dst
       this->pitch(),			  // dst_pitch
       data,				  // src
       sizeof(typename alloc::pixel_t) * this->dimV(),	  // src_pitch
       sizeof(typename alloc::pixel_t) * this->dimV(),	  // elm_size
       this->dimH(),			  // n_elm
       this->onGpu(),			  // dst_on_gpu
       on_gpu,			  // src_on_gpu
       syncId()			  // sync_id
       );
  }

  //* Copy a rectangle to the current frame from another ClrFrame
  template<class frm_alloc_t>
  void copyRectangleFromClrFrameAsync
  (const size_t bgn[2], //*< begin indices of rectangle in this frame
   ClrFrame<frm_alloc_t> &other
   ) {
    copyPitchAsync
      ((*this)[bgn[1]] + sizeof(typename alloc::pixel_t)*bgn[0], // dst
       this->pitch(),				   // dst_pitch
       other[0],				   // src
       other.getPitch(),			   // src_pitch
       sizeof(typename alloc::pixel_t)*other.dimV(),		   // elm_size
       other.dimH(),				   // n_elm
       this->onGpu(),				   // dst_on_gpu
       other.onGpu(),				   // src_on_gpu
       syncId()					   // sync_id
       );
  }

  void syncCopy() const {
#ifdef __NVCC_
    cudaStreamSynchronize(syncId());
#endif
  }

  void print() const {
    size_t v,h;
#ifdef __NVCC__
    if(onGpu()) {
      typename alloc::pixel_t *vec = new typename alloc::pixel_t[this->dimV()*this->dimH()];
      copyToVectorAsync(vec, false);
      blockSyncID();

      for (v=0; v<this->dimV(); v++) {
	printf("  ");
	for (h=0; h<this->dimH(); h++)
	  printf("%ld ", long (vec[h*this->dimV()+v]));
	printf("\n");
      }
      delete [] vec;
    }
    else
#endif
      for (v=0; v<this->dimV(); v++) {
	printf("  ");
	for (h=0; h<this->dimH(); h++)
	  printf("%ld ", long ((*this)[h][v]));
	printf("\n");
      }
  }

  size_t size() const { return this->dimV() * this->dimH(); }

private:
  typename ClrFrameSpec<alloc>::PixelArrayPtr *pxls_ptr;
  typename alloc::pixel_t *pxls;
  size_t ptch;			//*< pitch in bytes
#ifdef __NVCC__
  mutable cudaStream_t strm;
#endif

  friend class VidFrame<alloc>;
    
  void init(const ClrFrameSpec<alloc>& spec) {
    *static_cast<ClrFrameSpec<alloc> *>(this) = spec;
    pxls_ptr = spec.getPixelArrayPtr();
    pxls = **pxls_ptr;
    ptch = pxls_ptr->pitch();
  }

  void init(const size_t dm[2]) {
    init(ClrFrameSpec<alloc>(dm[0], dm[1]));
  }
  
};				// ClrFrame

class VidSpec
{
public:
  static const size_t max_colors = 3; // Maximum number of color components

  typedef size_t dim_spec_t[max_colors][2];
};

//* specification of a VidFrame format
template<typename alloc>
class VidFrameSpec
  : public VidSpec
{
public:
  VidFrameSpec(size_t n_c,	//*< number of colors actually used
	       const size_t dm[][2]	//*< Array of dimensions of size [n_c][2];
	       )
  { init(n_c, dm); }

  VidFrameSpec(const VidFrameSpec<alloc> &spec)
  { init(spec.n_clr, spec.clr_spec); }

  void print() const {
    printf("n_clr=%lu, dims:\n", (unsigned long) this->n_clr);
    for (size_t c=0; c<this->n_clr; c++)
      printf("[%lu,%lu]\n", (unsigned long) this->clr_spec[c].dimV(),
	     (unsigned long) this->clr_spec[c].dimH());
  }

  size_t nClr() const { return n_clr;}

  const dim_spec_t & getDims() const { return dims; }

  //* Computes offsets of color components.
  //* \param clr_ofsts - array of size nClr() returning the offsets of each
  //* color.
  //* \return the total size

  //* Get offset of color in pixel vector
  size_t clrOfst(size_t clr) const { return clr_ofst[clr]; }

  //* Get size of pixel_vector
  size_t size() const { return n_pxl; }

  static bool onGpu() { return alloc::onGpu(); }

  //* Allocate a 2D array of size \c nr by \c nc bytes. the (i,j) byte is at
  //* address:  base_addr + i + j*pitch. That is, data is stored column by
  //* column and each column is padded up to size \c pitch.
  //* \return pointer to beginning of the array
  static void *allocate(size_t n_r,    //*< number of rows in the array
			size_t n_c,    //*< number of columns in the array
			size_t *pitch //*< returns the pitch value
			)
  {
    *pitch = n_r;
    void * ret_val = new char[n_r * n_c];
    return ret_val;
  }
 
  //* deallocate space allocaed by allocate()
  static void deallocate(void *pxls)
  { delete [] (char *)pxls; }

private:
  
  size_t n_clr;			//* Number of colors
  ClrFrameSpec<alloc> clr_spec[max_colors];
  dim_spec_t dims;		//* dimension for each color (vertical,
				//* horizontal)
  size_t clr_ofst[max_colors];	//* offset of the pixels of each color from
				//* frame beginning
  size_t n_pxl;			//* total nubmer of pixels in the pixel vector
				//* for the frame.
  
  VidFrameSpec();

  void init(size_t n_c,	//*< number of colors actually used
	    const size_t dm[][2] //*< Array of dimensions of size [n_c][2];
	    ) {
    mex_assert(n_c <= this->max_colors,
               ("VidFrameSpec:Init", "Number of colors=%lu > %lu",
                (unsigned long)n_c, (unsigned long)this->max_colors ));
    
    this->n_clr = n_c;

    this->n_pxl = 0;
    size_t c;
    for(c=0; c<n_c; c++) {
      this->clr_spec[c].setDims(dm[c][0], dm[c][1]);
      this->dims[c][0] = dm[c][0];
      this->dims[c][1] = dm[c][1];
      this->clr_ofst[c] = this->n_pxl;
      this->n_pxl += dm[c][0] * dm[c][1];
    }
  }

  template<typename clr_alloc>
  void init(size_t n_c,	//*< number of colors actually used
	    const ClrFrameSpec<clr_alloc> clr_spc[] //*< Array of dimensions of size [n_c][2];
	    ) {
    dim_spec_t dm;
    for(size_t c=0; c<n_c; c++) {
      dm[c][0] = clr_spc[c].dimV();
      dm[c][1] = clr_spc[c].dimH();
    }
    init(n_c, dm);
  }

};				// VidFrameSpec
//* A video frame class
template<typename alloc>
class VidFrame
  : public GenericPoolObject
{
public:

  typedef typename alloc::pixel_t pixel_t;

  VidFrame(const VidFrameSpec<alloc> &spec, GenericObjectPool *pl = NULL)
    : GenericPoolObject(pl), frm_spec(spec)
  {
    for(size_t clr=0; clr<spec.nClr(); clr++)
      (*this)[clr].init(spec.getDims()[clr]);
  }

  ~VidFrame() { 
#ifdef __NVCC__
    gpuErrChk(cudaSuccess, "VidFrame:CudaStream_error", "");
#endif    
  }

  size_t nClr() const { return frm_spec.nClr();}

  ClrFrame<alloc>  & operator [] (size_t clr)
  { return cfrm[clr]; };

  const ClrFrame<alloc> & operator [] (size_t clr) const
  { return cfrm[clr]; };
  
  //* Copy out rectangles from each color frame to a vector asynchronously
  void copyRectanglesToVectorAsync
  (const size_t bgn[][2],	    //*< begin indices (V,H) per color
   const size_t end[][2],	    //*< one past end indices (V,H) per color
   typename alloc::pixel_t *data,		    //*< output vector
   bool on_gpu = false,		    //*< If true vector is on GPU
   SyncID sync_id = defaultSyncID
   ) const {
    for(size_t c=0; c<frm_spec.nClr(); c++)
      (*this)[c].copyRectangleToVectorAsync
	(bgn[c], end[c], data + clrOfst(c), on_gpu, sync_id);
  }

  //* Copy out rectangles from each color frame to a vector asynchronously,
  //* using different streams.
  void copyRectanglesToVectorAsync
  (const size_t bgn[][2],	    //*< begin indices (V,H) per color
   const size_t end[][2],	    //*< one past end indices (V,H) per color
   typename alloc::pixel_t *data,		    //*< output vector
   bool on_gpu,			    //*< If true vector is on GPU
   SyncID *psync_id		    //*< Array of nClr sync_idS
   ) const {
    for(size_t c=0; c<frm_spec.nClr(); c++)
      (*this)[c].copyRectangleToVectorAsync
	(bgn[c], end[c], data + clrOfst(c), on_gpu, psync_id[c]);
  }

  
  // Copy all color frames to a vector, using the syncId() of each of the color
  // frames.
  void copyToVectorAsync
  (typename alloc::pixel_t *data,		//*< output vector
   bool on_gpu = false		//*< If true vector is on GPU
   ) const {
    for(size_t c=0; c<frm_spec.nClr(); c++)
      (*this)[c].copyToVectorAsync(data + clrOfst(c), on_gpu);
  }

  //* Copy out rectangles from each color frame to another VidFrame,
  //* using the other frames' syncId()-s
  template<class frm_alloc_t>
  void copyRectangesToVidFrameAsync
  (const size_t bgn[][2],	    //*< begin indices (V,H) per color
   VidFrame<frm_alloc_t> &other //*< other frame
   ) const {
    for(size_t c=0; c<frm_spec.nClr(); c++)
      (*this)[c].copyRectangleToClrFrameAsync(bgn[c], other[c]);
  }
   
  //* Copy rectangles to each color frame from a vector asynchronously
  void copyRectanglesFromVectorAsync
  (const size_t bgn[][2],		  //*< begin indices
   const size_t end[][2],		  //*< one past end indices
   const typename alloc::pixel_t *data[],		  //*< input vector
   bool on_gpu = false		    //*< If true vector is on GPU
   ) {
    for(size_t c=0; c<frm_spec.n_clr(); c++)
      (*this)[c].copyRectangleFromVectorAsync
	(bgn[c],end[c], data + clrOfst(c), on_gpu); 
  }
  
  // Copy to all color frames from a vector
  void copyFromVectorAsync
  (const typename alloc::pixel_t *data,		//*< output vector
   bool on_gpu = false		//*< If true vector is on GPU
   ) {
    for(size_t c=0; c<frm_spec.nClr(); c++)
      (*this)[c].copyFromVectorAsync(data + clrOfst(c), on_gpu);
  }

  //* Copy in rectangles to each color frame from another VidFrame
  template<class frm_alloc_t>
  void copyRectangesFromVidFrameAsync
  (const size_t bgn[][2],	//*< begin indices (V,H) per color
   const VidFrame<frm_alloc_t> &other, //*< other frame
   SyncID sync_id = defaultSyncID
   ) {
    for(size_t c=0; c<frm_spec.nClr(); c++)
      (*this)[c].copyRectangleFromClrFrameAsync(bgn[c], other[c], sync_id);
  }

  void syncCopy() const {
    for(size_t c=1; c<frm_spec.nClr(); c++)
      (*this)[c].syncCopy();
  }
  
  void print() const {
    frm_spec.print();
    for (size_t c=0; c<frm_spec.nClr(); c++) {
      printf("Color %d:\n", int(c));
      (*this)[c].print();
     }
  }

  size_t clrOfst(size_t c) const { return frm_spec.clrOfst(c); }

  size_t size() const { return frm_spec.size(); }

private:
  const VidFrameSpec<alloc>  &frm_spec;
  ClrFrame<alloc> cfrm[VidFrameSpec<alloc> ::max_colors];

};				// VidFrame

#endif	/*  __VidFrame_H__ */
