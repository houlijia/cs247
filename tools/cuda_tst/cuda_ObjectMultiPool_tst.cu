#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <algorithm>
#include <list>

#include "CudaDevInfo.h"
#include "ObjectMultiPool.h"

class DemoVecSpec
{
public:
  // Constructors
  DemoVecSpec(size_t sz_ = 0)
    : sz(sz_) {}
  DemoVecSpec(const DemoVecSpec &other)
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
};				// DemoVecSpec

#if defined(__NVCC__)
class D_DemoVecSpec
  : public DemoVecSpec
{
public:
  D_DemoVecSpec(size_t sz_ = 0)
    : DemoVecSpec(sz_) {}
  D_DemoVecSpec(const DemoVecSpec & other)
    : DemoVecSpec(other) {}

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
	      "cuda_ObjectMultiPool:D_DemoVecSpec:alloc_error", "");
    return vec;
  }

  //* Deallocate vector allocated by allocate()
  static void deallocate(void * vec) {
    gpuErrChk(cudaFree(vec), "cuda_ObjectMultiPool:D_DemoVecSpec:alloc_error", "");
  }
};				// D_DemoVecSpec
#endif	// defined(__NVCC__)

//* A vector template class
template<typename data_t, class frm_spec_t>
class DemoVec : public GenericPoolObject
{
public:
  DemoVec(const frm_spec_t &spec_, GenericObjectPool * const pool)
    : GenericPoolObject(pool), spec(spec_), tst_indx(0), len(0), 
      data((data_t *) spec_.allocate(spec_.size()*sizeof(data_t), &on_gpu))
  {}

  ~DemoVec()
  { if(data != NULL) spec.deallocate(data); }

  unsigned tstIndx() const { return tst_indx; }
  void setTstIndx(unsigned val) { tst_indx = val; }
  
  size_t size() const {return spec.size();}
  bool onGpu() const { return on_gpu; }

  size_t length() const {return len; }
  void setLength(size_t val) {len = val; }

  operator data_t * () { return data; }
  operator const data_t * () const {return data; } 

private:
  const frm_spec_t &spec;
  unsigned tst_indx;		//*< an index for testing purposes;
  bool on_gpu;
  size_t len;
  data_t *data;
};

#ifdef __NVCC__
typedef D_DemoVecSpec spec_t;
#else
typedef DemoVecSpec spec_t;
#endif

typedef DemoVec<short, spec_t> vec_t;
typedef std::list<vec_t *> vec_list_t;
typedef  ObjectMultiPool<vec_t, spec_t> mpl_t ;
typedef  ObjectPool<vec_t, spec_t> pool_t ;

void print_status(const mpl_t &mpl, const vec_list_t vec_list) {
  printf("+++++++++++++++++\n");
  mpl.print();

  if(vec_list.empty())
    printf("vec_list is empty\n");
  else {
    printf("vec_list:");
    vec_list_t::const_iterator itr;
    unsigned k=0;
    for(itr = vec_list.begin();  itr != vec_list.end(); itr++) {
      const vec_t *p = *itr;
      printf("%s%4u%s 0x%06lX:0x%06lX", (k++%5)? ", ":",\n  ", p->tstIndx(),
	     p->onGpu()?"G":"C", (unsigned long) p->length(), (unsigned long)p->size());
    }
  }
  printf("\n------------------\n");
}

void help(const char *cmnd) {
  fprintf(stderr, "USAGE:\n"
	  "      %s [-v] <n_tst>  [<min_size> [<rnd_seed> [<max_size>]]]\n"
	  " The function tests the operation of ObjectMultiPool by running \n"
	  " <n_tst> (unsigned int) random test of get() or put() from or to the\n"
	  " MultiPool, respectively. The objects in the MultiPool are of class DemoVec\n"
	  " (defined here), which can be specified with any size from <min_size> to\n"
	  " <max_size>.\n"
	  " The default for <min_size> is 1 and for max_size it is 0x%08lX\n"
	  " In each test the function randomly either:\n"
	  "   A. Calls get() to get a DemoVec object of a random size and insers\n"
	  "      it into a FIFO.\n"
	  "   B. removes a DemoVec object from the FIFO and calls put() to put\n"
	  "      this object back into the MultiPool.\n"
	  " (if the FIFO is empty B is not done). Correctness is tested and the\n"
	  " status of the FIFO and the MultiPool are tested after each iteration.\n"
	  " If -v is specified status is printed after each iteration.\n"
	  , cmnd, (unsigned long)mpl_t::defaultMaxSize());
}

int main(int argc, const char *argv[]) {
  int err = EXIT_SUCCESS;
  unsigned n_tst;		//*< number of test to run
  unsigned i_tst;		//*< test index
  size_t min_size;		//*< minimum block size
  unsigned rnd_seed = 0;	//*< seed of random number generator
  unsigned long msz=1, mxsz = (unsigned long)mpl_t::defaultMaxSize();
  int n_opts = 0;
  bool verbose = false;
 
  if(argc <= 1) {
    help(argv[0]);
    return EXIT_SUCCESS;
  }

  if(!strcmp(argv[1], "-v")) {
    verbose = true;
    n_opts++;
  }
    
  if(argc-n_opts<2 || argc-n_opts>5 || sscanf(argv[n_opts+1], "%u", &n_tst) != 1 ||
     (argc-n_opts >= 3 && sscanf(argv[n_opts+2], "%lu", &msz) != 1) ||
     (argc-n_opts>=4 && sscanf(argv[n_opts+3], "%u", &rnd_seed) != 1) ||
     (argc-n_opts>=5 && sscanf(argv[n_opts+4], "%lu", &mxsz) != 1)
     ) {
    fprintf(stderr, "Illegal arguments\n\n");
    help(argv[0]);
    exit(EXIT_FAILURE);
  }

  size_t max_size = size_t(mxsz);

  min_size = size_t(msz);
  srand(rnd_seed);

  spec_t spec;
   
  vec_list_t vec_list;
  vec_list_t::iterator itr;
  size_t ttl_get = 0;
  unsigned last_get = 0;
  size_t max_vec_list = 0;
  mpl_t mpl(spec, min_size, max_size);

  clock_t start_time = clock();

     
  if(verbose) {
    printf("Initial state:\n");
    print_status(mpl, vec_list);
  }
    
  for(i_tst= 0; i_tst < n_tst;i_tst++ ) {
    // make the probability of get 7/16, slightly less than half. This makes
    // emptying vec_list have a probability of 1.
    unsigned rnd = rand();
    bool do_get = vec_list.empty() || ((rnd & 0xF) < 7);

    if(do_get) {
      size_t sz =
	size_t(floor(exp(double(rand())*log(double(max_size))/RAND_MAX)));
      if(verbose)
	printf("%u: get object of size %lu=0x%06lX\n",
	       i_tst, (unsigned long) sz, (unsigned long) sz);

      size_t ttl = mpl.total();
      
      vec_t & vec = mpl.get(sz);
      assert(vec.size() <= ((pool_t *)(vec.pool))->elmntSize());

      vec.setTstIndx(i_tst);
      vec.setLength(sz);
      vec_list.push_back(&vec);
      
      max_vec_list = std::max(vec_list.size(), max_vec_list);
      
      if(ttl == mpl.total()) last_get = ttl_get;
      ttl_get++;
    }
    else{
      unsigned indx = rand() % vec_list.size();
      bool use_discard = bool(rnd & 0x10);

      if(verbose)
	printf("%u %s object %u ", i_tst, use_discard?"discard":"put", indx);

      for(itr = vec_list.begin(); indx-- > 0; itr++);

      vec_t *pvec = *itr;

      if(verbose)
	printf("object index=%u, size=%lu=0x%06lX\n",
	       pvec->tstIndx(), (unsigned long)pvec->size(), (unsigned long)pvec->size());

      assert(pvec->size() <= ((pool_t *)(pvec->pool))->elmntSize());

      if(use_discard)
	pvec->discard();
      else
	mpl.put(*pvec);
      vec_list.erase(itr);				 
    }
    if(verbose)
      print_status(mpl, vec_list);
  }
	
  if(verbose) {
    printf("Status at end:\n");
    print_status(mpl, vec_list);
  }

  // Clear vec_list

  if(!verbose) {
    double elapsed = double(clock()-start_time)/CLOCKS_PER_SEC;
    printf("\n====== Done. in %f sec. (%f microsec/test) =====\n", elapsed, elapsed*1E6/n_tst);
  }
  else
    printf("\n====== Done. =====\n");
    
  if(verbose)
    printf("\n****** Clearing up ******\n\n");
  for(itr = vec_list.begin(); itr != vec_list.end();) {
    mpl.put(**itr);
    itr = vec_list.erase(itr);
    if(verbose)
      print_status(mpl, vec_list);
  }

  printf("Total gets: %lu, total blocks in pool: %lu, last_get: %u, max_vec_list=%lu\n",
	 (unsigned long) ttl_get, (unsigned long) mpl.total(), last_get,
	 (unsigned long)max_vec_list);


  print_status(mpl, vec_list);

  return err;
}
