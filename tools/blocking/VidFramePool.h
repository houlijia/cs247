/** \file
 * This file contains a class defining a pool of video frames
 */

#ifndef __VidFramePool_H__
#define __VidFramePool_H__

#include <stddef.h>
#include <stack>

using std::stack;

//* A pool of frame class objects. It is completely on the host
template<typename pixel_t, class frm_spec_t>
class VidFramePool: public frm_spec_t
{
public:
  //* Constructor
  VidFramePool(size_t n_c,	//*< number of colors actually used
	       size_t dm[][2]	//*< Array of dimensions of size [n_c][2];
	   )
    : frm_spec_t(n_c, dm) {}

  //* Constructor
  VidFramePool(const frm_spec_t &spec	//*< If true VidFrames are on GPU. Ignored if
	   )
    : frm_spec_t(spec) {}

  //* Destructor
  ~VidFramePool()  { while (!stk.empty()) delete &get(); }

  VidFrame<pixel_t, frm_spec_t> & get() {
    VidFrame<pixel_t, frm_spec_t> *pfrm;

    if(stk.empty()) {
      pfrm = new VidFrame<pixel_t, frm_spec_t>(*this);
      assert(pfrm != NULL);
    }
    else {
      pfrm = stk.top();
      stk.pop();
    }

     return *pfrm;
  }

  void put(VidFrame<pixel_t, frm_spec_t> &frm)
  { stk.push(&frm); }

  size_t size() const
  { return stk.size(); }


private:
  stack<VidFrame<pixel_t, frm_spec_t> *> stk;

};				// VidFramePool

#endif	/*  __VidFramePool_H__ */
