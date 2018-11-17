# include path
INCLUDE_DIRS:=. ../cuda_tst
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include ../util/defs.mak

ifneq ($(MEXCC),)

ROOT_NAMES:=real_dft_sort real_dft_unsort
WHT_NAMES:= wht

ifneq ($(GPU),NO)
ALL_G_NAMES:=$(patsubst %,cuda_%, $(ROOT_NAMES)) $(WHT_NAMES)
ALL_C_NAMES:=$(patsubst %,cc_%, $(ROOT_NAMES) wht)
MEX_CUDA_SRCS:=$(patsubst %,%_mex$(CU_EXT), $(ALL_G_NAMES)) 
MEX_GPU_FILES:=$(patsubst %,$(MEX_EXE_DIR)/%_mex$(MEX_EXT), $(ALL_G_NAMES))
else
ALL_C_NAMES:=$(patsubst %,cc_%, $(ROOT_NAMES)) $(WHT_NAMES)
endif

MEX_CPP_SRCS:= $(patsubst %,%_mex.cc, $(ALL_C_NAMES))
MEX_CPU_FILES:=$(patsubst %,$(MEX_EXE_DIR)/%_mex$(MEX_EXT), $(ALL_C_NAMES))

MEX_CPP_EXE_FILES:=$(MEX_CPU_FILES)
MEX_CUDA_EXE_FILES:=$(MEX_GPU_FILES)
MEX_EXE_FILES:=$(MEX_CPU_FILES) $(MEX_GPU_FILES)

$(MEX_EXE_FILES): $(MEX_LIB_DIR)/libtools$(MEX_L_EXT)

endif				# ifneq ($(MEXCC),)

# TEST FILES
TST_NAMES:=real_dft_sort wht

ifneq ($(GPU),NO)
TST_CUDA_SRCS:=$(patsubst %,cuda_%_tst$(CU_EXT), $(TST_NAMES))
TST_CUDA_EXE_FILES:=$(patsubst cuda_%_tst$(CU_EXT),$(TST_EXE_DIR)/cuda_%_tst$(EXE_EXT), $(TST_CUDA_SRCS))
$(TST_CUDA_EXE_FILES): $(TST_LIB_DIR)/libgputools$(LIB_EXT)
endif

TST_CPP_SRCS:= $(patsubst %,cc_%_tst.cc, $(TST_NAMES))
TST_CPP_EXE_FILES:=$(patsubst cc_%_tst.cc,$(TST_EXE_DIR)/cc_%_tst$(EXE_EXT), $(TST_CPP_SRCS))
$(TST_CPP_EXE_FILES): $(TST_LIB_DIR)/libtools$(LIB_EXT)

include ../util/rules.mak

