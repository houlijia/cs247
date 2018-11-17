# include path
INCLUDE_DIRS:=. ../cuda_tst
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include ../util/defs.mak

ifneq ($(MEXCC),)

# Mex function that call the entry point
ifneq ($(GPU),NO)
MEX_CUDA_SRCS:=cuda_quant_mex$(CU_EXT)
MEX_GPU_FILES:=$(MEX_EXE_DIR)/cuda_quant_mex$(MEX_EXT)
else
MEX_CPP_SRCS:=cc_quant_mex.cc
MEX_CPU_FILES:=$(MEX_EXE_DIR)/cc_quant_mex$(MEX_EXT)
endif

endif 				# ifneq ($(MEXCC),)

# TEST FILES
ifneq ($(GPU),NO)
TST_CUDA_SRCS:=cuda_quant_tst$(CU_EXT)
TST_CUDA_EXE_FILES:=$(TST_EXE_DIR)/cuda_quant_tst$(EXE_EXT)
$(TST_CUDA_EXE_FILES):  $(TST_LIB_DIR)/libgputools$(LIB_EXT)
endif

ifneq ($(MEXCC),)

MEX_CPP_EXE_FILES:=$(MEX_CPU_FILES)
MEX_CUDA_EXE_FILES:=$(MEX_GPU_FILES)
MEX_EXE_FILES:=$(MEX_CPU_FILES) $(MEX_GPU_FILES)
$(MEX_EXE_FILES): $(MEX_LIB_DIR)/libtools$(MEX_L_EXT)

endif		# ifneq ($(MEXCC),)

TST_EXE_FILES:=$(TST_CPP_EXE_FILES) $(TST_CUDA_EXE_FILES)

include ../util/rules.mak

