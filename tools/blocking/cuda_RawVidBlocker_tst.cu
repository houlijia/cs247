#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>

#include "CudaDevInfo.h"
#include "VidFrame.h"
#include "ObjectPool.h"
#include "ObjectMultiPool.h"
#include "print_vec.h"
#include "RawVidBlocker.h"
#include "fast_heap.h"

static unsigned verbose = 0;
static unsigned n_frms;
static int pixel_type = 'c';
static bool frm_gpu;			//*< True if frms are on GPU
static size_t frm_dim[3][2];
static unsigned n_clr = 1;
static bool blk_gpu;	                //*< True if blocks are on GPU
static size_t blk_sz[3][3];		//*< blk_sz[c] contains the block dimensions for color c
static size_t blk_olp[3][3];		//*< blk_olp[c] contains the block overlaps for color c
static unsigned n_t_blks = 0;		//*< If 0, read single
					//*blocks. Otherwise, read all the
					//*blocks of \c n_t_blks temporal
					//*blocks at a time.


static void help(const char *cmnd) {
  printf("Run as:\n"
	 "    %s [-v<vrbs>] <n_frms> <frm_spec> <blk_spec> [<n_t_blks>]\n\n",
	 cmnd);
  printf
    ("The program creates a RawVidBlocker object, vid_blkr. Then it generates \n"
     "frames and inserts them into vid_blkr. After each frame is inserted the\n"
     "program checks if blocks can be extracted. If yes, the program extracts\n"
     "all available blocks and then removes the frames that will not be\n"
     "necessary for future blocks. In verbose mode, the frames and blocks\n"
     "are printed out. The entries in the frames are increasing by 1\n"
     "with wrap around, which allows for easy correctness checking.\n\n"
     );

  printf
     ("ARGUMENTS:\n"
      "  -v<vrbs>: Specifies the verbosity of the program (0-2, default = 0)\n"
      "  <n_frms>: has the format <nf>[<pixel_type>] where <nf> is the (positive) number\n"
      "            of frames to process and <pixel_type> is an optional single letter\n"
      "            which specifies the type of the frame pixels and can get the\n"
      "            values of:\n"
      "              'c': - unsigned char (defalut)\n"
      "              'h': - unsigned short\n"
      "              'i': - unsigned int\n"
      "              'l': - unsigned long\n"
      "  <frm_spec>: has the format [<P>]<v>,<h>[/[<rv>,<rh>], where <P> is a single\n"
      "            letter indicating where the frames are stored - 'C' for CPU\n"
      "            (default) and 'G' for GPU; <v>,<h> are the vertical and\n"
      "            horizontal dimensions of the frame If <h> is not followed by '/'\n"
      "            then there is a single Y component. Otherwise there are\n"
      "            YUV components and the <v>,<h> refer to the Y component. If\n"
      "            the '/' is followed by '<rv>,<rh>', then these are the\n"
      "            the respective vertical and horizontal ratios between the Y\n"
      "            dimensions and the U,V dimensions. The default ratio is 1.\n"
      "  <blk_spec.: has the format [<P>]<v>,<h>,<t>[:<ov>,<oh>,<ot>], where <P>\n"
      "            is a single letter indicating where the frames are stored - 'C' for\n"
      "             CPU (default) and 'G' for GPU; <v>,<h>,<t>\n"
      "            are the vertical, horizontal and temporal dimensions of a\n"
      "            block, respectively, and <ov>,<oh>,<ot>, if present are the\n"
      "            corresponding block overlaps in each dimension (0 overlap\n"
      "            if <ov>,<oh>,<ot> are not specified).\n"
      "  <n_t_blks>: number of temporal blocks to extract at once. If specified and\n"
      "            positive, the program extracts vectors of  <n_t_blks> temporal\n"
      "            blocks at once. Default = 0\n"
      );
}

static bool chk_gpu(char *arg, char **parg) {
  switch(arg[0]) {
  case 'C': *parg = arg+1; return false;
  case 'G':
#ifdef __NVCC__
    *parg = arg+1; return true;
#else
    fprintf(stderr,
	    "The program was not compiled with nvcc. frames/blocks cannot be specified on GPU\n");
    exit(EXIT_FAILURE);
#endif
  default: *parg = arg; return false;
  }
}

static void parse_args(int argc, char *argv[]) {
  int n_opts = 0;
  int nrd;
  char *ptr;
  bool fail = false;
  char *arg;
  unsigned dm[2], rt[3]={1,1,1};
  unsigned c,d;
  char ch;
  unsigned bsz[3], bolp[3]={0,0,0};

  if(sscanf(argv[1], "-v%u", &verbose))
     n_opts++;
 
  if(argc <= 3+n_opts) {
    fprintf(stderr, "not enough argments.\nRun %s without arguments for help\n",
	    argv[0]);
    exit(EXIT_FAILURE);
  }

  if(argc > 5+n_opts) {
    fprintf(stderr, "Too many argments.\nRun %s without arguments for help\n",
	    argv[0]);
    exit(EXIT_FAILURE);
  }

  // Parse n_frms
  n_frms = (unsigned) strtoul(argv[1+n_opts], &ptr, 0);
  if(!n_frms)
    fail = true;
  else if(ptr[0] != '\0') {
    pixel_type = ptr[0];
    fail = (ptr[1] != '\0') || (strchr("chil", pixel_type) == NULL);
  }
  if(fail) {
    fprintf(stderr,
	    "Illegal value for <n_frms>.\nrun %s without arguments for help\n",
	    argv[0]);
    exit(EXIT_FAILURE);
  }
  
  // parse <frm_spec>
  frm_gpu = chk_gpu(argv[2+n_opts], &arg);
  nrd = sscanf(arg, "%u,%u%c%u,%u", &dm[0], &dm[1], &ch, &rt[0], &rt[1]);
  switch(nrd) {
  case 2:
    fail = (!dm[0] || !dm[1]);
    n_clr = 1; frm_dim[0][0] = dm[0]; frm_dim[0][1] = dm[1];
    break;
  case 3:
    fail = (!dm[0] || !dm[1]) || (ch != '/');
    if(fail) break;
    n_clr = 3;
    frm_dim[0][0] = frm_dim[1][0] = frm_dim[2][0] = dm[0];
    frm_dim[0][1] = frm_dim[1][1] = frm_dim[2][1] = dm[1];
    break;
  case 5:
    fail = (!dm[0] || !dm[1]) || (ch != '/') || (!rt[0] || !rt[1]) ||
      (dm[0] % rt[0]) || (dm[1] % rt[1]);
    if(fail) break;
    n_clr = 3;
    frm_dim[0][0] = dm[0]; frm_dim[0][1] = dm[1];
    frm_dim[1][0] = frm_dim[2][0] = dm[0]/rt[0];
    frm_dim[1][1] = frm_dim[2][1] = dm[1]/rt[1];
    break;
  default:
    fail = true; break;
  }
  if(fail) {
    fprintf(stderr,
	    "Illegal value for <frm_spec>.\nrun %s without arguments for help\n",
	    argv[0]);
    exit(EXIT_FAILURE);
  }
  
  // parse blk_spec
  blk_gpu = chk_gpu(argv[3+n_opts], &arg);
  if(strchr(arg, ':'))
    fail = sscanf(arg, "%u,%u,%u:%u,%u,%u",
		  &bsz[0] , &bsz[1] , &bsz[2], &bolp[0], &bolp[1], &bolp[2]
		  ) != 6;
  else
    fail = sscanf(arg, "%u,%u,%u",
		  &bsz[0] , &bsz[1] , &bsz[2]
		  ) != 3;
  for(d=0; d<3; d++) {
    fail = fail || bolp[d]>=bsz[d] || (bsz[d]%rt[d]) || (bolp[d]%rt[d]);
    if(fail) break;
    blk_sz[0][d] = bsz[d]; blk_olp[0][d] = bolp[d];
    for(c=1; c<n_clr; c++) {
      blk_sz[c][d] = bsz[d]/rt[d];
      blk_olp[c][d] = bolp[d]/rt[d];
    }
  }

  if(fail) {
    fprintf(stderr,
	    "Illegal value for <blk_spec>.\nrun %s without arguments for help\n",
	    argv[0]);
    exit(EXIT_FAILURE);
  }

  // parse n_t_blks;
  if(argc > 4+n_opts) {
    n_t_blks = (unsigned) strtoul(argv[4+n_opts], &ptr, 0);
    if(*ptr != '\0') {
      fprintf(stderr,
	      "Illegal value for <n_t_blks>.\nrun %s without arguments for help\n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  
}

static void print_args() {
  unsigned c;

  printf("\nPARSED ARGUMENTS:\n");
  printf("Processing %u frames. Pixel type is unsigned %s\n", n_frms, 
	 pixel_type == 'c'? "char":
	 pixel_type == 'h'? "short":
	 pixel_type == 'i'? "int": "long");
  printf("Frms on %s, n_clr=%u, dims (VxH): ", frm_gpu?"GPU":"Host", n_clr);
  if(n_clr==1)
    printf("[%u,%u]\n", (unsigned)frm_dim[0][0], (unsigned)frm_dim[0][1]);
  else {
    printf("[");
    for(c=0; c<n_clr; c++)
      printf("[%u,%u]%s", (unsigned)frm_dim[c][0], (unsigned)frm_dim[c][1], 
	     c==n_clr-1? "]\n":",");
  }

  printf("Blocks on %s. Number of temporal blocks to read at once: %u\n",
	 blk_gpu?"GPU":"Host", n_t_blks);
  for(c=0; c<n_clr; c++)
    printf("clr %u: blk_size: [%4u,%4u,%4u], overlap: [%4u,%4u,%4u]\n", c,
	   unsigned(blk_sz[c][0]), unsigned(blk_sz[c][1]), unsigned(blk_sz[c][2]),
	   unsigned(blk_olp[c][0]), unsigned(blk_olp[c][1]), unsigned(blk_olp[c][2]));
  printf("--------------------------------------------------------------------\n");
}

template<typename pixel_t>
const char *pixel_fmt() {return " "; }

template<> const char *pixel_fmt<unsigned char>()  { return " %3hu"; }
template<> const char *pixel_fmt<unsigned short>() { return " %5hu"; }
template<> const char *pixel_fmt<unsigned int>()   { return " %8u"; }
template<> const char *pixel_fmt<unsigned long>()  { return " %10hu"; }

template<typename pixel_t>
GenericHeapElement  &allocateBlkSpace(unsigned len) {
  size_t sz = len*sizeof(pixel_t);
  
#ifdef __NVCC__
  if(blk_gpu)
    return d_fast_heap->get(sz);
  else
    return h_fast_heap->get(sz);

#else

  return fast_heap->get(sz);

#endif
}

template<typename alloc>
static void run_tests()
{
  typedef VidFrameSpec<alloc> frm_spec_t;
  typedef typename alloc::pixel_t pixel_t;
  
  frm_spec_t frm_spec(n_clr, frm_dim); //*< Frame specifier

  size_t blk_t_size = blk_sz[0][2];   //*< temporal block size
  size_t blk_t_ovlp = blk_olp[0][2];  //*< temporal block overlap
  unsigned clr;			//*< Index variable for loops on color
  size_t blk_s_size[3][2];	//*< spatial block size
  size_t blk_s_ovlp[3][2];	//*< spatial block overlap
  unsigned nxt_t_blk = 0;	//*< Index of the next temporal block to
				//*process
  unsigned h;			//*< horizontal block index
  unsigned v;			//*< vertical block index


  //* Set arrays for RawVidBlocker
  for(clr=0; clr<n_clr; clr++) {
    blk_s_size[clr][0] = blk_sz[clr][0];  blk_s_size[clr][1] = blk_sz[clr][1];
    blk_s_ovlp[clr][0] = blk_olp[clr][0]; blk_s_ovlp[clr][1] = blk_olp[clr][1];
  }
  
  //* The RawVidBlocker object 
  RawVidBlocker<alloc> vblkr(frm_spec, blk_s_size, blk_s_ovlp,
					   blk_t_size, blk_t_ovlp);

  if(verbose) {
    printf("\nVidBlocker:\n");
    vblkr.printSpec();
    printf("--------------------------------------------------------------------\n");
    
    printf("VidBlocker status: ");
    vblkr.printStatus();
  }

  unsigned frm_len = vblkr.lengthFrm();
  pixel_t * const frm_vec = new pixel_t[frm_len]; //*< Vector of frame pixel

  //*values
  pixel_t *blk_vec, *blk_vec_end;
  GenericHeapElement *pghe;
  if(n_t_blks)
    pghe = &allocateBlkSpace<pixel_t>(n_t_blks * vblkr.frmBlkLength());
  else
    pghe = &allocateBlkSpace<pixel_t>(vblkr.blkLength());

  GenericHeapElement &pblk_vec = *pghe;
  blk_vec = (pixel_t *) *pblk_vec;
  
 //* pointers to color beginnings
  pixel_t * const frm_clr_vec[3] =
    {frm_vec,
     frm_vec + vblkr.lengthVFrm(0) *vblkr.lengthHFrm(0),
     frm_vec + vblkr.lengthVFrm(0) *vblkr.lengthHFrm(0) +
     vblkr.lengthVFrm(1) *vblkr.lengthHFrm(1)
    };

  pixel_t frm_val = 0;		//*< Value of next pixel in frame

  for(unsigned ifrm=0; ifrm<n_frms; ifrm++) {
    unsigned k;

    // Fill up frm_vec
    for(k=0; k<frm_len; k++) frm_vec[k] = frm_val++;

#if 0
    if(verbos e >=2) {
      printf("Frame %4u values:\n", ifrm);
      print_vec(frm_len, frm_vec, 10, pixel_fmt<pixel_t>());
    }
#endif

    // Insert frame in vblkr
    vblkr.insertFwd(1, frm_clr_vec);

    if(verbose>=2) {
      printf("Frame %4u:\n", ifrm);
      vblkr.lastFrame().print();
    }

    if(verbose) {
      printf("VidBlocker status: ");
      vblkr.printStatus();
    }

    if(n_t_blks > 0) {
      for(;vblkr.tBlkOffset() <= nxt_t_blk &&
	    vblkr.tBlkOffset()+vblkr.nTBlks() >= nxt_t_blk+n_t_blks;
	  nxt_t_blk += n_t_blks)
	{
	  vblkr.getFrmBlks(n_t_blks, nxt_t_blk, blk_vec, &blk_vec_end, blk_gpu);
	  assert(blk_vec_end - blk_vec == std::ptrdiff_t(n_t_blks * vblkr.frmBlkLength()));

	  if(verbose > 0) {
	    printf("temporal blocks %u -%u, vector length = %lu\n",
		   nxt_t_blk, nxt_t_blk+n_t_blks-1, (unsigned long)(blk_vec_end-blk_vec));

	    if(verbose > 1) {
	      const pixel_t *vec = blk_vec;
	      size_t sz, total=0;
	      for(size_t t=0; t<n_t_blks; t++)
		for(h=0; h < vblkr.nHBlks(); h++)
		  for(v=0; v < vblkr.nVBlks(); v++) {
		    sz = vblkr.blkLength(v,h);
		    printf("block (%u,%u,%lu), length=%lu\n", 
			   v, h, (unsigned long)(t+nxt_t_blk), (unsigned long) sz);
		    print_vec(sz, vec, 10, pixel_fmt<pixel_t>(), blk_gpu);
		    vec += sz;
		    total += sz;
		  }
	      assert(total == (unsigned long)(blk_vec_end-blk_vec));
	    }
	  }
	}
    }
    else {
      for(;vblkr.tBlkOffset() <= nxt_t_blk &&
	    vblkr.tBlkOffset()+vblkr.nTBlks() >= nxt_t_blk+1;
	  nxt_t_blk++)
	for(h=0; h < vblkr.nHBlks(); h++)
	  for(v=0; v < vblkr.nVBlks(); v++) {
	    vblkr.getBlk(v,h,nxt_t_blk, blk_vec, &blk_vec_end, blk_gpu);
	    assert(blk_vec_end - blk_vec == std::ptrdiff_t(vblkr.blkLength(v,h)));

	    if(verbose > 0) {
	      printf("block (%u,%u,%u), length=%u\n", v, h, nxt_t_blk, 
		     unsigned(blk_vec_end-blk_vec));
	      if(verbose > 1)
		print_vec(blk_vec_end-blk_vec, blk_vec, 10,
			  pixel_fmt<pixel_t>(), blk_gpu);
	    }
	  }
    }
    vblkr.removeFrmsBeforeTBlk(nxt_t_blk);

  }

#if 0
  if(verbose >= 2) {
    // Removing and inserting
    vblkr.removeFrms(1, frm_clr_vec, false);
    printf("First frame values:\n");
    print_vec(frm_len, frm_vec, 10, pixel_fmt<pixel_t>());

    vblkr.insertBwd(1, frm_clr_vec);
    printf("First frame:\n");
    vblkr.firstFrame().print();
  }
#endif

  pblk_vec.discard(); 
  delete [] frm_vec;
}


int main(int argc, char *argv[])  {
 
  if(argc == 1) {
    help(argv[0]);
    exit(EXIT_SUCCESS);
  }
  
  parse_args(argc, argv);
  if(verbose) print_args();

#ifdef __NVCC__
  if(frm_gpu) {
    switch(pixel_type) {
    case 'c': run_tests<D_RectAlloc<unsigned char> >(); break;
    case 'h': run_tests<D_RectAlloc<unsigned short> >(); break;
    case 'i': run_tests<D_RectAlloc<unsigned int> >(); break;
    case 'l': run_tests<D_RectAlloc<unsigned long> >(); break;
    }
  }
  else {
    switch(pixel_type) {
    case 'c': run_tests<H_RectAlloc<unsigned char> >(); break;
    case 'h': run_tests<H_RectAlloc<unsigned short> >(); break;
    case 'i': run_tests<H_RectAlloc<unsigned int> >(); break;
    case 'l': run_tests<H_RectAlloc<unsigned long> >(); break;
    }
  }
#else
  switch(pixel_type) {
  case 'c': run_tests<RectAlloc<unsigned char> >(); break;
  case 'h': run_tests<RectAlloc<unsigned short> >(); break;
  case 'i': run_tests<RectAlloc<unsigned int> >(); break;
  case 'l': run_tests<RectAlloc<unsigned long> >(); break;
  }
#endif

  printf("test done\n");

  (void) GPU_PREV_ERR_CHK();

  reset_fast_heap();
  
  return 0;
}
