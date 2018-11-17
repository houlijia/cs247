# include path
INCLUDE_DIRS:=.
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include ../util/defs.mak

MEX_CPP_SRCS:=
MEX_CUDA_SRCS:=
TST_CPP_SRCS:=
TST_CUDA_SRCS:=

# Specify libraries
ifneq ($(MEXCC),)
MEX_TOOLS_LIB:=$(MEX_LIB_DIR)/libtools$(MEX_L_EXT)
$(MEX_TOOLS_LIB): $(patsubst %,$(MEX_OBJ_DIR)/%$(MEX_O_EXT), mex_context mex_tools timers)
endif

TST_TOOLS_LIB:=$(TST_LIB_DIR)/libtools$(LIB_EXT)
$(TST_TOOLS_LIB): $(patsubst %,$(TST_OBJ_DIR)/%$(OBJ_EXT), timers fast_heap)

ifneq ($(GPU),NO)

ifneq ($(MEXCC),)
$(MEX_TOOLS_LIB): $(patsubst %,$(MEX_OBJ_DIR)/%$(MEX_O_EXT), CudaDevInfo \
  cuda_fast_heap mex_gpu_tools)
endif

TST_GPU_TOOLS_LIB:=$(TST_LIB_DIR)/libgputools$(LIB_EXT)
$(TST_GPU_TOOLS_LIB): $(patsubst %,$(TST_OBJ_DIR)/%$(OBJ_EXT), timers CudaDevInfo \
  cuda_fast_heap cuda_vec_do )

else

ifneq ($(MEXCC),)
$(MEX_TOOLS_LIB): $(patsubst %,$(MEX_OBJ_DIR)/%$(MEX_O_EXT), fast_heap )
endif

endif

ifneq ($(MEXCC),)
MEX_LIB_FILES:=$(MEX_TOOLS_LIB)
endif
TST_LIB_FILES:=$(TST_TOOLS_LIB) $(TST_GPU_TOOLS_LIB)

ifneq ($(MEXCC),)

# Specify MEX files
ifneq ($(GPU),NO)

M_SRC_ROOTS:=init sub_sqr mean mean_stdv

# Mex function that call the entry point
M_SRC_NAMES:=$(patsubst %,%_mex, $(patsubst %,cuda_%,$(M_SRC_ROOTS)) \
    reset_fast_heap init_mex_context delete_mex_context mexGPU)
MEX_GPU_FILES:=$(patsubst %,$(MEX_EXE_DIR)/%$(MEX_EXT), $(M_SRC_NAMES))

MEX_CUDA_SRCS+=$(patsubst %, %$(CU_EXT), $(M_SRC_NAMES))
MEX_CUDA_SRCS+=$(patsubst %,%$(CU_EXT), cuda_fast_heap \
  CudaDevInfo mex_context mex_gpu_tools)

MEX_CPP_SRCS+= mex_tools.cc timers.cc

else

M_SRC_ROOTS:=sub_sqr mean mean_stdv

M_SRC_NAMES:=$(patsubst %,%_mex, $(patsubst %,cc_%,$(M_SRC_ROOTS)) \
    reset_fast_heap init_mex_context delete_mex_context mexGPU)

MEX_CPU_FILES:=$(patsubst %,$(MEX_EXE_DIR)/%$(MEX_EXT), $(M_SRC_NAMES))
MEX_CPP_SRCS+=$(patsubst %, %.cc, $(M_SRC_NAMES))
MEX_CPP_SRCS+=$(patsubst %, %.cc, fast_heap mex_context mex_tools timers)

endif

MEX_CPP_EXE_FILES:=$(MEX_CPU_FILES)
MEX_CUDA_EXE_FILES:=$(MEX_GPU_FILES)
MEX_EXE_FILES:=$(MEX_CPU_FILES) $(MEX_GPU_FILES)

$(MEX_EXE_FILES): $(MEX_TOOLS_LIB)

endif				# ifneq ($(MEXCC),)

# TEST FILES
ifneq ($(GPU),NO)
TG_SRC_ROOTS:=$(patsubst %,%_tst,$(patsubst %,cuda_%,init sub_sqr vec mean ObjectMultiPool \
    fast_heap) cumsum)
TST_CUDA_EXE_FILES:=$(patsubst %,$(TST_EXE_DIR)/%$(EXE_EXT),$(TG_SRC_ROOTS)) 

TST_CUDA_SRCS+=$(patsubst %,%$(CU_EXT),$(TG_SRC_ROOTS))
TST_CUDA_SRCS+=$(patsubst %,%$(CU_EXT),cuda_fast_heap CudaDevInfo cuda_vec_do )

$(TST_CUDA_EXE_FILES) : $(TST_GPU_TOOLS_LIB)

endif

TST_CPP_EXE_FILES:=$(patsubst %,$(TST_EXE_DIR)/cc_%_tst$(EXE_EXT), ObjectMultiPool fast_heap)
TST_CPP_SRCS+=$(patsubst %,cc_%_tst.cc, ObjectMultiPool fast_heap)
TST_CPP_SRCS+=fast_heap.cc
TST_CPP_SRCS+=timers.cc

$(TST_CPP_EXE_FILES): $(TST_TOOLS_LIB)


include ../util/rules.mak


