/** \file
* This file contains the class RawVidBlocker, which facilitates storing of
* video frames and extracting blocks from them
*/

#ifndef __RawVidBlocker_h__
#define __RawVidBlocker_h__

#include <limits.h>
#include <float.h>
#include <deque>

#if defined(MATLAB_MEX_FILE) && MATLAB_MEX_FILE 
#include "mex_tools.h"

#if !defined printf
#define printf mexPrintf
#endif

#endif

#include "VidFrame.h"

//* Generic virtual basis for RawVidBlocker, to use when the pixel_t and/or
//* GPU usage are determined at run time

class GenericRawVidBlocker
{
public:
  GenericRawVidBlocker() {}
  virtual ~GenericRawVidBlocker() {}

  virtual void printSpec() const = 0;

  virtual void printStatus() const = 0;

  //* \return the number of colors
  virtual unsigned nColors() const = 0; 

  //* \return true if frames are on GPU
  virtual bool onGPU() const = 0;

  //* \return vertical length of frame at a color \c c
  virtual unsigned lengthVFRM(unsigned c) const = 0;

  //* \return horizontal length of frame at a color \c c
  virtual unsigned lengthHFRM(unsigned c) const = 0;

#if defined(MATLAB_MEX_FILE) && MATLAB_MEX_FILE
  //* \return class ID of pixel
  virtual mxClassID pixelClassID() const = 0;
#endif

  //* \return the number of saved frames
  virtual unsigned frmsCOUNT() const = 0;
  
  //* Insert \c nfr video frames at the end.
  virtual
  void insertFWD(unsigned nfr,	//*< Number of frames to insert
		 const void * const vid[], //* Pointers to the data for each color
		 bool on_gpu = false //*< If true vectors are on GPU 
		 ) = 0;

  //* Insert \c nfr video frames at the beginning.
  virtual
  void insertBWD(unsigned nfr,	//*< Number of frames to insert
		 const void * const vid[], //* Pointers to the data for each color
		 bool on_gpu = false //*< If true vectors are on GPU 
		 ) = 0;

  //* Remove \c nfr video frames
  virtual
  void removeFrms(unsigned nfr	//*< Number of frames to remove
		  ) = 0;
  
  //* Remove \c nfr video frames and return
  //* them in \c vid. When the frames are no longer needed, each frame should
  //* be disposed of by calling discard().
  virtual
  void removeFrms(unsigned nfr,	//*< Number of frames to remove
		  //* An array of nfr pointers for the returned VidFrame objects.
		  void * const vid[],
		  bool on_gpu = false //*< If true vectors are on GPU 
		  ) = 0;
  
  //* Remove all video frames preceding temporal block \c t_blk
  virtual
  void removeFrmsBeforeTBlk(size_t t_blk) = 0;
   
  //* Remove \c nfr video frames preceding temporal block \c t_blk and return
  //* them in \c vid. When the frames are no longer needed, each frame should
  //* be disposed of by calling discard().
  virtual
  void removeFrmsBeforeTBlk(size_t t_blk,
			    //* An array of nfr pointers for returned VidFrame objects.
			    void * const vid[],
			    bool on_gpu = false //*< If true vectors are on GPU 
			    ) = 0;
  
  //* Same as \c removeFrmsBeforeTBlk(t_blk), but t_blk is relative to first block
  //* in buffer
  virtual
  void removeFrmsBeforeTBlkRel(unsigned t_blk) = 0;
   
  //* Same as \c removeFrmsBeforeTBlk(t_blk, vid, on_gpu), but t_blk is relative 
  //* to first block in buffer
  virtual
  void removeFrmsBeforeTBlkRel(unsigned t_blk,
			       //* Array of nfr pointers for returned VidFrame objects.
			       void * const vid[],
			       bool on_gpu = false //*< If true vectors are on GPU 
			       ) = 0;

 //* \return the size of the block at spatial position \c (v,h).
  virtual unsigned blkLENGTH(unsigned v, unsigned h) const = 0;

  //* \return the number of elements in a vector of blocks for a whole frame
  virtual unsigned frmBlkLENGTH() const = 0;

  //* Read a block into a vector.
  virtual
  void getBlk(unsigned v,	  //*< vertical block index
	      unsigned h,	  //*< horizontal block index
	      size_t t,		  //*< temporal block index
	      void *vec,	  //*< output vector
	      void **vec_end,  //*< If not NULL returns a pointer
	                          //* to one past end of copied vector
	      bool on_gpu = false //*< If true vector is on GPU
	      ) const = 0;

  //* Read frame blocks into a vector
  virtual
  void getFrmBlks(unsigned n_fr,  //*< number of frame blocks 
		  size_t t,	  //*< temporal first block index
		  void *vec,	  //*< output vector
		  void **vec_end, //*< If not NULL returns a pointer
				//* to one past end of copied vector
		  bool on_gpu = false //*< If true vector is on GPU
		  ) const = 0;

  //* Same as getBlk, but the temporal block index \c t is relative to the
  //* first block in the buffer.
  virtual
  void getBlkRel(unsigned v,	     //*< vertical block index
		 unsigned h,	     //*< horizontal block index
		 unsigned t,	     //*< temporal block index relative to
				     //* \c tBlkOffset()
		 void *vec,	     //*< output vector
		 void **vec_end,  //*< If not NULL returns a pointer
				     //* to one past end of copied vector
		 bool on_gpu = false //*< If true vector is on GPU
		 ) const = 0;

  //* Same as getBlk, but the temporal block index \c t is relative to the
  //* first block in the buffer.
  virtual
  void getFrmBlksRel(unsigned n_fr,      //*< number of frame blocks 
		     unsigned t,	 //*< temporal first block index
				         //* relative to \c tBlkOffset()
		     void *vec,	 //*< output vector
		     void **vec_end,  //*< If not NULL returns a pointer
				         //* to one past end of copied vector
		     bool on_gpu = false //*< If true vector is on GPU
		     ) const = 0;

  //* \return a string containing the pixel type (e.g. "int").
  template <typename pixel_t>
  static const char *pixelType();


#if defined(MATLAB_MEX_FILE) && MATLAB_MEX_FILE
  //* \return class ID of pixel
  virtual mxClassID pixelCLASSID() const = 0;
#endif

};				// GenericRawVidBlocker

//* This function is implemented through template specialization. The
//* default here is merely a place holder.
template<typename pixel_t>
const char * GenericRawVidBlocker::pixelType()
{ return "unknown"; } 

template<>
const char * GenericRawVidBlocker::pixelType<long double>()
{ return "long double"; } 

template<>
const char * GenericRawVidBlocker::pixelType<double>()
{ return "double"; } 

template<>
const char * GenericRawVidBlocker::pixelType<float>()
{ return "float"; } 

template<>
const char * GenericRawVidBlocker::pixelType<unsigned char>()
{ return "unsigned char"; } 

template<>
const char * GenericRawVidBlocker::pixelType<unsigned short>()
{ return "unsigned short"; } 

template<>
const char * GenericRawVidBlocker::pixelType<unsigned int>()
{ return "unsigned int"; } 

template<>
const char * GenericRawVidBlocker::pixelType<unsigned long>()
{ return "unsigned long"; } 

template<>
const char * GenericRawVidBlocker::pixelType<signed char>()
{ return "signed char"; } 

template<>
const char * GenericRawVidBlocker::pixelType<short>()
{ return "short"; } 

template<>
const char * GenericRawVidBlocker::pixelType<int>()
{ return "int"; } 

template<>
const char * GenericRawVidBlocker::pixelType<long>()
{ return "long"; } 


//* Template class for saving frames and extracting block pixel vectors.
template<typename alloc>
class RawVidBlocker : public virtual GenericRawVidBlocker
{
public:
  typedef VidFrame<alloc> vid_frm_t;
  typedef VidFrameSpec<alloc> frm_spec_t;
  typedef typename alloc::pixel_t pixel_t;

  RawVidBlocker(frm_spec_t &spec, //*< Frame specification
		size_t b_s_sz[][2], //*< block spatial size (v,h), per color
		size_t b_s_olp[][2], //*< block spatial offset (v,h) per color
		size_t b_t_sz,	     //*< block temporal size
		size_t b_t_olp	     //*< block temporal offset
		);

  virtual ~RawVidBlocker();

  virtual void printSpec() const;

  virtual void printStatus() const;
  
  typedef pixel_t * const *vid_ptr_t;
  typedef const pixel_t * const *const_vid_ptr_t;

  //* Insert \c nfr video frames at the end.
  void insertFwd(unsigned nfr,	//*< Number of frames to insert
		 const_vid_ptr_t vid, //* Pointers to the data for each color
		 bool on_gpu = false //*< If true vectors are on GPU 
		 );

  //* Virtual version of insertFwd
  void insertFWD(unsigned nfr,	//*< Number of frames to insert
		 const void * const vid[], //* Pointers to the data for each color
		 bool on_gpu = false //*< If true vectors are on GPU 
		 )
  { insertFwd(nfr, (const_vid_ptr_t)(vid), on_gpu); }
  
  //* Insert \c nfr video frames at the beginning.
  void insertBwd(unsigned nfr,	//*< Number of frames to insert
		 const_vid_ptr_t vid, //* Pointers to the data for each color
		 bool on_gpu = false //*< If true vectors are on GPU 
		 );

  //* Virtual version of insertBwd
  void insertBWD(unsigned nfr,	//*< Number of frames to insert
		 const void * const vid[], //* Pointers to the data for each color
		 bool on_gpu = false //*< If true vectors are on GPU 
		 )
  { insertBwd(nfr, (const_vid_ptr_t)(vid), on_gpu); }

  //* Remove \c nfr video frames (virtual)
  void removeFrms(unsigned nfr	//*< Number of frames to remove
		  );
  
  //* Remove \c nfr video frames and return
  //* them in \c vid. When the frames are no longer needed, each frame should
  //* be disposed of by calling discard().
  void removeFrms(unsigned nfr,	//*< Number of frames to remove
		  //* An array of nfr pointers for the returned VidFrame objects.
		  vid_ptr_t vid,
		  bool on_gpu = false //*< If true vectors are on GPU 
		  );

  //* virtual version or removeFrms
  void removeFrms(unsigned nfr,	//*< Number of frames to remove
		  //* An array of nfr pointers for the returned VidFrame objects.
		  void * const vid[],
		  bool on_gpu = false //*< If true vectors are on GPU 
		  )
  { removeFrms(nfr, vid_ptr_t(vid), on_gpu); }

  //* Remove all video frames preceding temporal block \c t_blk
  void removeFrmsBeforeTBlk(size_t t_blk)
  { removeFrms(nFrmsBeforeTBlk(t_blk)); }
   
  //* Remove \c nfr video frames preceding temporal block \c t_blk and return
  //* them in \c vid. When the frames are no longer needed, each frame should
  //* be disposed of by calling discard().
  void removeFrmsBeforeTBlk(size_t t_blk,
			    //* An array of nfr pointers for returned VidFrame objects.
			    vid_ptr_t vid,
			    bool on_gpu = false //*< If true vectors are on GPU 
			    )
  { removeFrms(nFrmsBeforeTBlk(t_blk), vid, on_gpu); }

  //* virtual version of removeFrmsBeforeTBlk()
  void removeFrmsBeforeTBlk(size_t t_blk,
			    //* An array of nfr pointers for returned VidFrame objects.
			    void * const vid[],
			    bool on_gpu = false //*< If true vectors are on GPU 
			    )
  { removeFrmsBeforeTBlk(t_blk, vid_ptr_t(vid), on_gpu); }

  
  //* Same as \c removeFrmsBeforeTBlk(t_blk), but t_blk is relative to first block
  //* in buffer
  void removeFrmsBeforeTBlkRel(unsigned t_blk)
  { removeFrms(nFrmsBeforeTBlkRel(t_blk)); }
   
  //* Same as \c removeFrmsBeforeTBlk(t_blk, vid, on_gpu), but t_blk is relative 
  //* to first block in buffer
  void removeFrmsBeforeTBlkRel(unsigned t_blk,
			       //* Array of nfr pointers for returned VidFrame objects.
			       vid_ptr_t vid,
			       bool on_gpu = false //*< If true vectors are on GPU 
			       )
  { removeFrms(nFrmsBeforeTBlkRel(t_blk), vid, on_gpu); }


  void removeFrmsBeforeTBlkRel(unsigned t_blk,
			       //* Array of nfr pointers for returned VidFrame objects.
			       void * const vid[],
			       bool on_gpu = false //*< If true vectors are on GPU 
			       )
  { removeFrmsBeforeTBlkRel(t_blk, vid_ptr_t(vid), on_gpu); }


  //* \return a reference to the k-th pointer (with bound checking)
  vid_frm_t & operator [] (unsigned k) { return *frms.at(k); }

  //* \return a const reference to the k-th pointer. An error is thrown if out of bounds
  const vid_frm_t & operator [] (unsigned k) const { return *frms.at(k); }

  //* \return a reference to the first frame (with emptiness checking)
  vid_frm_t & firstFrame() { return (*this)[0]; }
  
  //* \return a const reference to the first frame (with emptiness checking)
  const vid_frm_t & firstFrame() const { return (*this)[0]; }
  
  //* \return a reference to the last frame (with emptiness checking)
  vid_frm_t & lastFrame() { return (*this)[frmsCount()-1]; }
  
  //* \return a const reference to the last frame (with emptiness checking)
  const vid_frm_t & lastFrame() const { return (*this)[frmsCount()-1]; }
  
  //* \return the saved frames temporal offset (index of first stored frame)
  size_t frmsOffset() const {return frms_ofst; }

  //* \return the number of saved frames
  unsigned frmsCount() const { return unsigned(frms.size()); }

  //* Same as frmsCount() but virtual. \return the number of saved frames
  unsigned frmsCOUNT() const { return frmsCount(); }

  //* \return the number of colors
  unsigned nCOLORS() const { return (unsigned) frms_pool.nClr(); }

  //* Same as nCOLORS but virtual. \return the number of colors
  virtual unsigned nColors() const { return nCOLORS(); }
  
  //* \return true if frames are on GPU
  bool onGpu() const { return frms_pool.onGpu(); }

#if defined(MATLAB_MEX_FILE) && MATLAB_MEX_FILE
  //* \return class ID of pixel
  mxClassID pixelClassID() const { return mx_class_id<pixel_t>(); }

  //* \return class ID of pixel
  //* virtual version of \c pixelClassID()
  virtual mxClassID pixelCLASSID() const { return pixelClassID(); }
#endif

  //* Same as nGpu but virtual. \return true if frames are on GPU
  bool onGPU() const { return onGpu(); }

  //* \return vertical length of frame at a color \c c
  unsigned lengthVFrm(unsigned c) const { return unsigned(frms_pool.getDims()[c][0]); }

  //* Same as lengthVFrm() but virtual. \return vertical length of frame at a color \c c
  virtual unsigned lengthVFRM(unsigned c) const { return lengthVFrm(c); }

  //* \return horizontal length of frame at a color \c c
  unsigned lengthHFrm(unsigned c) const { return unsigned(frms_pool.getDims()[c][1]); }

  //* Same as lengthHrm() but virtual. \return horizontal length of frame at a color \c c
  virtual unsigned lengthHFRM(unsigned c) const { return lengthHFrm(c); }

  //* \return total number of pixel in a frame
  unsigned lengthFrm() const;

  //* \return the number of blocks vertically
  unsigned nVBlks() const { return blk_cnt[0]; }

  //* \return the number of blocks horizontally
  unsigned nHBlks() const { return blk_cnt[1]; }

  //* \return the number of spatial blocks in a frame
  unsigned nBlksInFrm() const { return blk_vh_cnt; }

  //* \return the vertical offset of block \c v, color \c c
  unsigned blkVOffset(unsigned c, unsigned v) const
  { return (v? ((unsigned)v * blk_incr[c][0] - slack[c][0]) : 0); }

  //* \return the horizontal offset of block \c h, color \c c
  unsigned blkHOffset(unsigned c, unsigned h) const
  { return (h? ((unsigned)h * blk_incr[c][1] - slack[c][1]) : 0); }

  //* \return the vertical length of block \c v, color \c c
  unsigned blkVLength(unsigned c, unsigned v) const;
  
  //* \return the horizontal length of block \c h, color \c c
  unsigned blkHLength(unsigned c, unsigned h) const;
  
 //* \return the size of the color block of color \c c at spatial position
  //* \c (v,h)
  unsigned clrBlkLength(size_t c, unsigned v, unsigned h) const
  { unsigned cc = unsigned(c);
    return unsigned(blkVLength(cc,v) * blkHLength(cc,h) * blkTLength());
  }

  // Returns the temporal size of a block.
  unsigned blkTLength() const { return blk_t_sz; }

  // Returns the temporal overlap of a block.
  unsigned blkTOverlap() const { return blk_t_ovlp; }

  //* \return the maximum color block size for color \c c
  unsigned clrBlkLength(unsigned c) const;
  
  //* \return the size of the block at spatial position \c (v,h)
  unsigned blkLength(unsigned v, unsigned h) const;

  //* \return the size of the block at spatial position \c (v,h). 
  //* Same as blkLength(v,h), but virtual
  virtual unsigned blkLENGTH(unsigned v, unsigned h) const
  { return blkLength(v,h); }

  //* \return the maximum block size
  unsigned blkLength() const;

  //* \return the number of elements in a vector of blocks for a whole frame
  unsigned frmBlkLength() const;

  //* \return the number of elements in a vector of blocks for a whole frame
  //* Same as frmBlkLength() but virtual
  virtual unsigned frmBlkLENGTH() const
  { return frmBlkLength(); }
  
  //* \return the index of the first temporal block in the buffer
  size_t tBlkOffset() const
  { return (frmsOffset() + (blk_t_sz-blk_t_ovlp) - 1)/blk_t_incr; }

  //* \return the number of temporal blocks in the buffer
  unsigned nTBlks() const;

  //* \return the number of frames preceding temporal block \c t_blk in the buffer
  unsigned nFrmsBeforeTBlk(size_t t_blk) {
    assert(t_blk >= tBlkOffset());
    return nFrmsBeforeTBlkRel(unsigned(t_blk - tBlkOffset()));
  }

  //* \return the number of frames preceding relative temporal block \c t_blk in the buffer
  unsigned nFrmsBeforeTBlkRel(unsigned t_blk //*< temporal block index relative to
				             //*\c tBlkOffset()
			   )
  { return std::min(frmsCount(), t_blk * blk_t_incr + blk_t_frm_ofst); }

  //* Read a block into a vector.
  void getBlk(unsigned v,	  //*< vertical block index
	      unsigned h,	  //*< horizontal block index
	      size_t t,		  //*< temporal block index
	      pixel_t *vec,	  //*< output vector
	      pixel_t **vec_end,  //*< If not NULL returns a pointer
	                          //* to one past end of copied vector
	      bool on_gpu = false //*< If true vector is on GPU
	      ) const {
    assert(t >= tBlkOffset());
    getBlkRel(v, h, unsigned(t-tBlkOffset()), vec, vec_end, on_gpu);
  }

  //* virtual version of getBlk()
  void getBlk(unsigned v,	  //*< vertical block index
	      unsigned h,	  //*< horizontal block index
	      size_t t,		  //*< temporal block index
	      void *vec,	  //*< output vector
	      void **vec_end,  //*< If not NULL returns a pointer
	                          //* to one past end of copied vector
	      bool on_gpu = false //*< If true vector is on GPU
	      ) const
  { getBlk(v,h,t, (pixel_t *)(vec), (pixel_t **)(vec_end), on_gpu); }

  //* Read frame blocks into a vector
  void getFrmBlks(unsigned n_fr,      //*< number of frame blocks 
		  size_t t,	      //*< temporal first block index
		  pixel_t *vec,	      //*< output vector
		  pixel_t **vec_end,  //*< If not NULL returns a pointer
		                      //* to one past end of copied vector
		  bool on_gpu = false //*< If true vector is on GPU
		  ) const {
    assert(t >= tBlkOffset());
    getFrmBlksRel(n_fr, unsigned(t-tBlkOffset()), vec, vec_end, on_gpu);
  }

  //* virtual version of getFrmBlks()
  virtual
  void getFrmBlks(unsigned n_fr,      //*< number of frame blocks 
		  size_t t,	      //*< temporal first block index
		  void *vec,	      //*< output vector
		  void **vec_end,  //*< If not NULL returns a pointer
		                      //* to one past end of copied vector
		  bool on_gpu = false //*< If true vector is on GPU
		  ) const
  { getFrmBlks(n_fr, t, (pixel_t *)(vec), (pixel_t **)(vec_end), on_gpu); }

  //* Same as getBlk, but the temporal block index \c t is relative to the
  //* first block in the buffer.
  void getBlkRel(unsigned v,	     //*< vertical block index
		 unsigned h,	     //*< horizontal block index
		 unsigned t,	     //*< temporal block index relative to
				     //* \c tBlkOffset()
		 pixel_t *vec,	     //*< output vector
		 pixel_t **vec_end,  //*< If not NULL returns a pointer
				     //* to one past end of copied vector
		 bool on_gpu = false //*< If true vector is on GPU
		 ) const;

  //* virtual version of getBlkRel()
  virtual
  void getBlkRel(unsigned v,	     //*< vertical block index
		 unsigned h,	     //*< horizontal block index
		 unsigned t,	     //*< temporal block index relative to
				     //* \c tBlkOffset()
		 void *vec,	     //*< output vector
		 void **vec_end,  //*< If not NULL returns a pointer
				     //* to one past end of copied vector
		 bool on_gpu = false //*< If true vector is on GPU
		 ) const
  { getBlkRel(v,h,t, (pixel_t *)(vec), (pixel_t **)(vec_end), on_gpu); }

  //* Same as getBlk, but the temporal block index \c t is relative to the
  //* first block in the buffer.
  void getFrmBlksRel(unsigned n_fr,      //*< number of frame blocks 
		     unsigned t,	 //*< temporal first block index
				         //* relative to \c tBlkOffset()
		     pixel_t *vec,	 //*< output vector
		     pixel_t **vec_end,  //*< If not NULL returns a pointer
				         //* to one past end of copied vector
		     bool on_gpu = false //*< If true vector is on GPU
		     ) const;

  //* virtual version of getFrmBlksRel()
  virtual
  void getFrmBlksRel(unsigned n_fr,      //*< number of frame blocks 
		     unsigned t,	 //*< temporal first block index
				         //* relative to \c tBlkOffset()
		     void *vec,	 //*< output vector
		     void **vec_end,  //*< If not NULL returns a pointer
				         //* to one past end of copied vector
		     bool on_gpu = false //*< If true vector is on GPU
		     ) const
  { getFrmBlksRel(n_fr, t, (pixel_t *)(vec), (pixel_t **)(vec_end), on_gpu); }

private:
  typedef std::deque<vid_frm_t *> frms_t;
  
  ObjectPool<vid_frm_t, frm_spec_t> frms_pool;
  frms_t frms;
  size_t frms_ofst;

  //* number of blocks along (vertical, horizontal) axes
  unsigned blk_cnt[2];		
  unsigned blk_vh_cnt;		//*< number of spatial blocks
  
  //* Vertical and horizontal block size per color
  unsigned blk_sz[frm_spec_t::max_colors][2]; 

  //* Vertical and horizontal block overlap per color
  unsigned blk_ovlp[frm_spec_t::max_colors][2];

  unsigned blk_t_sz;		//*< number of temporal frames in a block
  unsigned blk_t_ovlp;		//*< temporal block overlap (frames)
  unsigned blk_t_incr;		//*< temporal step between blocks
  unsigned blk_t_frm_ofst;	//*< number of frames in the buffer
				//*before beginning of first block

  //* Vertical and horizontal block increment per color
  unsigned blk_incr[frm_spec_t::max_colors][2];
  
  unsigned slack[frm_spec_t::max_colors][2]; //*< Slack of vertical blocks

  static unsigned cmpAXisBlkCnt(unsigned frm_len, //*< Frame length along axis
				unsigned blk_sz,	//*< Block length along axis
				unsigned ovlp	//*< Block overlap along axis
				)
  {
    // ceil of (frm_len-ovlp)/(blk_sz-ovlp)
    return unsigned((frm_len-ovlp + (blk_sz-ovlp) - 1)/(blk_sz-ovlp));
  }

 //* Read a block into a vector, asynchoronously. The function assumes
  //* that frames are available frms array.
  void getBlkAsync(unsigned v,	  //*< vertical block index
		   unsigned h,	  //*< horizontal block index
		   unsigned t,	  //*< temporal block index in the
		                  //*frms array, i.e. absolute index
		                  //*minus frms_ofst
		   pixel_t *vec,  //*< output vector
		   pixel_t **vec_end, //*< If not NULL returns a pointer
		                      //* to one past end of copied vector
		   bool on_gpu = false //*< If true vector is on GPU
		   ) const;
  
  //* Read a color block into a vector, asynchoronously. The function assumes
  //* that frames are available frms array
  void getClrBlkAsync(unsigned c,	  //*< color
		      unsigned v,	  //*< vertical block index
		      unsigned h,	  //*< horizontal block index
		      unsigned t,	  //*< temporal block index in the
					  //*frms array, i.e. absolute index
					  //*minus frms_ofst
		      pixel_t *vec,	  //*< output vector
		      pixel_t **vec_end,  //*< If not NULL returns a pointer
					  //* to one past end of copied vector
		      bool on_gpu = false //*< If true vector is on GPU
		      ) const;

 };				// class RawVidBlocker

template<typename alloc> 
RawVidBlocker<alloc>::RawVidBlocker(frm_spec_t &spec,
				    size_t b_s_sz[][2],
				    size_t b_s_olp[][2],
				    size_t b_t_sz,	    
				    size_t b_t_olp
				    )
  : frms_pool(spec), 
    frms_ofst(0),
    blk_t_sz(unsigned(b_t_sz)),
    blk_t_ovlp(unsigned(b_t_olp)),
    blk_t_incr(unsigned(blk_t_sz - b_t_olp)),
    blk_t_frm_ofst(0)
{ 
  unsigned clr;
  unsigned dim;

  for(dim=0; dim<2; dim++) {
    if(spec.getDims()[0][dim] <= b_s_sz[0][dim])
      blk_cnt[dim] = 1;
    else {
      unsigned step = unsigned(b_s_sz[0][dim] - b_s_olp[0][dim]);
      blk_cnt[dim] = unsigned((spec.getDims()[0][dim] - b_s_olp[0][dim] + step - 1) / step);
    }
  
    for(clr=0; clr<spec.nClr(); clr++) {
      blk_sz[clr][dim] = unsigned(b_s_sz[clr][dim]);
      blk_ovlp[clr][dim] = unsigned(b_s_olp[clr][dim]);
      blk_incr[clr][dim] = unsigned(b_s_sz[clr][dim] - b_s_olp[clr][dim]);
      
      slack[clr][dim] = unsigned(( b_s_olp[clr][dim] + blk_cnt[dim] * blk_incr[clr][dim] -
			  spec.getDims()[clr][dim] 
				   ) / 2);
    }
  }
  blk_vh_cnt = blk_cnt[0] * blk_cnt[1];
} 

template<typename alloc> 
RawVidBlocker<alloc>::~RawVidBlocker() {
  while(!frms.empty()) {
    frms.front()->discard();	// Return first element to the pool
    frms.erase(frms.begin());	// Remove first element
  }
}

template<typename alloc> 
void RawVidBlocker<alloc>::printStatus()  const {
  printf("first frame=%lu, number of frames=%u. first t-blk=%lu, number of t-blks=%u\n",
	 (unsigned long)frmsOffset(), frmsCount(), (unsigned long)tBlkOffset(), nTBlks());
}

template<typename alloc> 
void RawVidBlocker<alloc>::printSpec()  const {
  const char *pixel_t_str = this->pixelType<pixel_t>();

  printf("Pixels are %s. %u colors.\n", pixel_t_str, nCOLORS());
  printf("Frames on %s. Frames dimensions per color (VxH):", onGpu()? "GPU":"CPU");

  unsigned c,k;

  for(c=0; c<nCOLORS(); c++)
    printf(" %ux%u", lengthVFrm(c), lengthHFrm(c));
  printf("\n");

  printf("%u x %u = %u blocks, temporal size: %u frames, overalp: %u frames\n",
	 nVBlks(), nHBlks(), nBlksInFrm(), blkTLength(), blkTOverlap());

  for(c=0; c<nCOLORS(); c++) {
    printf("Color %u blocks:\n", c);
    printf("Vert. block size:  ");
    for(k=0; k<nVBlks(); k++) printf(" %3u", blkVLength(c,k));
    printf("\nVert. block offset:");
    for(k=0; k<nVBlks(); k++) printf(" %3u", blkVOffset(c,k));
    printf("\nHorz. block size:  ");
    for(k=0; k<nHBlks(); k++) printf(" %3u", blkHLength(c,k));
    printf("\nHorz. block offset:");
    for(k=0; k<nHBlks(); k++) printf(" %3u", blkHOffset(c,k));
    printf("\n");
  }
}

template<typename alloc> 
void RawVidBlocker<alloc>::insertFwd
(unsigned nfr,
 const pixel_t * const vid[],
 bool on_gpu
 ) {
  unsigned fr_cnt = frmsCount();
  unsigned c;			//*< index for loops over colors
  const pixel_t *vd[frm_spec_t::max_colors]; //*< Modifiable copy of vid

  for(c=0; c<nCOLORS(); c++) vd[c] = vid[c];
  
  while(nfr-- > 0) {
    vid_frm_t * pfrm = &frms_pool.get();
    for(c = 0; c< nCOLORS(); c++) {
      (*pfrm)[c].copyFromVectorAsync(vd[c], on_gpu);
      vd[c] += (*pfrm)[c].size();
    }
    frms.push_back(pfrm);
  }

  for(size_t k = fr_cnt; k<frmsCount(); k++)
    frms[k]->syncCopy();
}

template<typename alloc> 
void RawVidBlocker<alloc>::insertBwd
(unsigned nfr,
 const pixel_t * const vid[],
 bool on_gpu
 ) {
  assert(nfr <= frms_ofst);
  
  unsigned c;			//*< index for loops over colors
  const pixel_t *vd[frm_spec_t::max_colors]; //*< Modifiable copy of vid

  unsigned k;

  for(c=0; c<nCOLORS(); c++) vd[c] = vid[c];
  
  for(k=0; k<nfr; k++)
    frms.push_front(&frms_pool.get());

  for(k=0; k<nfr; k++) { 
    for(c = 0; c< nCOLORS(); c++) {
      vid_frm_t &frm = *(frms[k]);
      frm[c].copyFromVectorAsync(vd[c], on_gpu);
      vd[c] += frm[c].size();
    }
  }

  frms_ofst -= nfr;
  blk_t_frm_ofst = (blk_t_frm_ofst + nfr) % blk_t_incr;
  
  for(k=0; k<nfr; k++)
    frms[k]->syncCopy();
}

template<typename alloc>
void RawVidBlocker<alloc>::removeFrms(unsigned nfr)
{
  if(nfr == 0) return;
  assert(nfr <= frmsCount());
  frms_ofst += nfr;
  blk_t_frm_ofst =
    (blk_t_frm_ofst + blk_t_incr - (nfr % blk_t_incr)) % blk_t_incr;

  while(nfr-- > 0) {
    frms_pool.put(firstFrame());
    frms.pop_front();
  }
}

template<typename alloc>
void RawVidBlocker<alloc>::removeFrms(unsigned nfr, 
						    pixel_t * const vid[],
						    bool on_gpu
)
{
  if(nfr == 0) return;
  assert(nfr <= frmsCount());
  frms_ofst += nfr;
  blk_t_frm_ofst =
    (blk_t_frm_ofst + blk_t_incr - (nfr % blk_t_incr)) % blk_t_incr;

  unsigned k;			//*< index for loops over frames
  unsigned c;			//*< index for loops over colors
  pixel_t *vd[frm_spec_t::max_colors]; //*< Modifiable copy of vid

  if(nfr == 0) return;
  for(c=0; c<nCOLORS(); c++) vd[c] = vid[c];

  for(k=0; k<nfr; k++) {
     for(c = 0; c< nCOLORS(); c++) {
       (*this)[k][c].copyToVectorAsync(vd[c], on_gpu);
       vd[c] += (*this)[k][c].size();
     }
  }

  while(nfr-- > 0) {
    this->firstFrame().syncCopy(); // Wait for copying to complete
    // Return first element to pool
    frms_pool.put(this->firstFrame());
    frms.pop_front();
  }
}

template<typename alloc>
unsigned RawVidBlocker<alloc>::lengthFrm() const {
  unsigned val=lengthVFrm(0) * lengthHFrm(0);
  for(unsigned c=1; c<nCOLORS(); c++) val += lengthVFrm(c) * lengthHFrm(c);
  return val;
}

template<typename alloc>
unsigned RawVidBlocker<alloc>::blkVLength(unsigned c, unsigned v) const
{
  return (v==0?( blk_sz[c][0] - slack[c][0]): 
	  v==(nVBlks()-1)? (lengthVFrm(c) - blkVOffset(c,v)):
	  blk_sz[c][0]);
}


template<typename alloc>
unsigned RawVidBlocker<alloc>::blkHLength(unsigned c, unsigned h) const
{
  return (h==0?( blk_sz[c][1] - slack[c][1]): 
	  h==(nHBlks()-1)? (lengthHFrm(c) - blkHOffset(c,h)):
	  blk_sz[c][1]);
}

 
template<typename alloc>
unsigned RawVidBlocker<alloc>::clrBlkLength(unsigned c) const
{
  return 
    (nVBlks()>1? blk_sz[c][0]: lengthVFrm(c)) *
    (nHBlks()>1? blk_sz[c][1]: lengthHFrm(c)) *
    blk_t_sz;
}

template<typename alloc>
unsigned RawVidBlocker<alloc>::blkLength(unsigned v, unsigned h) const
{
  unsigned s = clrBlkLength(0,v,h);
  for (unsigned c=1; c<nCOLORS(); c++)
    s += clrBlkLength(c,v,h);
  return s;
}

template<typename alloc>
unsigned RawVidBlocker<alloc>::blkLength() const
{
  unsigned s = clrBlkLength(0);
  for (unsigned c=1; c<nCOLORS(); c++)
    s += clrBlkLength(c);
  return s;
}

template<typename alloc>
unsigned RawVidBlocker<alloc>::frmBlkLength() const
{
  unsigned s = 0;
  
  for(unsigned h=0; h<nHBlks(); h++)
    for(unsigned v=0; v<nVBlks(); v++)
      s += blkLength(v,h);
	
  return s;
}

template<typename alloc>
unsigned RawVidBlocker<alloc>::nTBlks() const
{
  unsigned rslt;

  if(frmsCount() >= blk_t_sz)
    rslt = unsigned((frmsOffset() + frmsCount() - blk_t_ovlp)/blk_t_incr - tBlkOffset());
  else
    rslt = 0;
  return rslt;
}

template<typename alloc>
void RawVidBlocker<alloc>::getBlkRel(unsigned v,
						   unsigned h,
						   unsigned t,
						   pixel_t *vec,
						   pixel_t **vec_end,
						   bool on_gpu
						   ) const
{
  assert(nTBlks() > t);

  unsigned tbgn = t * blk_t_incr + blk_t_frm_ofst;
  unsigned tend = tbgn + blk_t_sz;

  getBlkAsync(v, h, tbgn, vec, vec_end, on_gpu);

  // Synchronize with each frame from which we copied
  for(unsigned t = tbgn; t != tend; t++)
    frms[t]->syncCopy();
    
}

template<typename alloc>
void RawVidBlocker<alloc>::getFrmBlksRel(unsigned n_fr,
						       unsigned t,
						       pixel_t *vec,
						       pixel_t **vec_end,
						       bool on_gpu
						       ) const
{
  assert(nTBlks() >= t+n_fr);

  unsigned tbgn =  t * blk_t_incr + blk_t_frm_ofst;
  unsigned tend = tbgn + blk_t_ovlp + n_fr*blk_t_incr;

  for(unsigned tfr = tbgn; n_fr-- > 0; tfr += blk_t_incr)
    for(unsigned h=0; h<nHBlks(); h++)
      for(unsigned v=0; v<nVBlks(); v++)
	getBlkAsync(v, h, tfr, vec, &vec, on_gpu);

  if(vec_end != NULL)
    *vec_end = vec;
  
 // Synchronize with each frame from which we copied
  for(unsigned t = tbgn; t != tend; t++)
    frms[t]->syncCopy();
}

template<typename alloc>
void RawVidBlocker<alloc>::getBlkAsync(unsigned v,
						     unsigned h,
						     unsigned t,
						     pixel_t *vec,
						     pixel_t **vec_end,
						     bool on_gpu
						     ) const
{
  for(unsigned c=0; c<nCOLORS(); c++)
    getClrBlkAsync(c,v,h,t,vec, &vec, on_gpu);
  
  if(vec_end != NULL)
    *vec_end = vec;
}

template<typename alloc>
void RawVidBlocker<alloc>::getClrBlkAsync(unsigned c,
							unsigned v,
							unsigned h,
							unsigned t,
							pixel_t *vec,
							pixel_t **vec_end,
							bool on_gpu
							) const 
{
  unsigned blen[2] = { blkVLength(c,v), blkHLength(c,h)};
  size_t bbgn[2] = {blkVOffset(c,v), blkHOffset(c,h)};
  size_t bend[2] = {bbgn[0] + blen[0], bbgn[1] + blen[1]};

  for(unsigned tfr=t; tfr != t+blk_t_sz; tfr++) {
    const vid_frm_t &frm = *frms[tfr];
    frm[c].copyRectangleToVectorAsync(bbgn, bend, vec, on_gpu, frm[c].syncId());
    vec += blen[0]*blen[1];
  }

  if(vec_end != NULL)
    *vec_end = vec;
}

#endif  /* __RawVidBlocker_h__ */
