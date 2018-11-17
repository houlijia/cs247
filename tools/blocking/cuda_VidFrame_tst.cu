#include <stdio.h>

#include "print_vec.h"
#include "RawVidBlocker.h"

int main(int argc, char *argv[])  {
  (void) argc;
  (void) argv;

  size_t fdm[3][2] = {{6,8}, {3,4}, {3,4}};
  typedef RectAlloc<short> rect_alloc_t;
  VidFrameSpec<rect_alloc_t> frm_spec(sizeof(fdm)/sizeof(fdm[0]), fdm);


  VidFrame<rect_alloc_t> frm(frm_spec);
  size_t frm_size = frm.size();
  printf("frm_spec: (size=%lu)\n", (unsigned long)frm_size);
  frm_spec.print();
  (void) GPU_PREV_ERR_CHK();

  short *vec = new short[frm_size];
  size_t k;
  for(k=0; k<frm_size; k++)
    vec[k] = short(k);
  printf("vec=\n");
  print_vec(frm_size, vec, 10, " %6hd", false);

  frm.copyFromVectorAsync(vec);
  (void) GPU_PREV_ERR_CHK();
  frm.syncCopy();
 
  printf("frm is:\n");
  frm.print();

  delete [] vec;
  printf("\n----------------\n");


  size_t bgn[][2]={{0,1}, {0,0}, {0,0}};
  size_t end[][2]={{2,3}, {1,3}, {2,1}};

  size_t gdm[3][2];
  for (size_t c=0; c < frm.nClr(); c++) {
    gdm[c][0] = end[c][0] - bgn[c][0];
    gdm[c][1] = end[c][1] - bgn[c][1];
  }

#ifdef __NVCC__
  typedef D_RectAlloc<short> alloc_t ;
#else
  typedef RectAlloc<short> alloc_t;
#endif

  typedef VidFrameSpec<alloc_t > vid_frm_spec_t;
  typedef VidFrame<alloc_t> vid_frm_t;
  typedef ObjectPool<vid_frm_t, vid_frm_spec_t> pool_t;

  pool_t pool(vid_frm_spec_t(sizeof(gdm)/sizeof(gdm[0]), gdm));
  (void) GPU_PREV_ERR_CHK();

  for(int rpt=0; rpt<3; rpt++) {
    printf("*** repetition %d pool size is %lu\n", rpt, (unsigned long)pool.size());

    vid_frm_t & gr = pool.get();
    
    frm.copyRectangesToVidFrameAsync(bgn, gr);
    gr.syncCopy();
    
    printf("gr:\n");
    gr.print();

    GenericHeapElement *ppgvec = &fast_heap->get(sizeof(vid_frm_t::pixel_t)*gr.size());
    vid_frm_t::pixel_t *gvec = static_cast<vid_frm_t::pixel_t *>(**ppgvec);
    gr.copyToVectorAsync(gvec, false);
    gr.syncCopy();

    printf("gvec:\n");
    print_vec(gr.size(), gvec, 10, " %6hd", false);
    ppgvec->discard();
    
     pool.put(gr);
  }

  (void) GPU_PREV_ERR_CHK();

  printf("***** Done! *****\n");
  return 0;
}
