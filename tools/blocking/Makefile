# include path
INCLUDE_DIRS:=. ../cuda_tst
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include ../util/defs.mak

MEX_CPP_SRCS:=
MEX_CUDA_SRCS:=
TST_CPP_SRCS:=
TST_CUDA_SRCS:=

ifneq ($(GPU),NO)
G_TST_SRCS:= cuda_VidFrame_tst$(CU_EXT) cuda_RawVidBlocker_tst$(CU_EXT)
TST_CUDA_SRCS+=$(G_TST_SRCS)
endif
C_TST_SRCS:= cc_VidFrame_tst.cc cc_RawVidBlocker_tst.cc
TST_CPP_SRCS+=$(C_TST_SRCS)

VBLKR_SRC_NAMES:=initRawVidBlocker_mex deleteRawVidBlocker_mex insertRawVidBlocker_mex \
  removeRawVidBlocker_mex getRawVidBlocker_mex

ifneq ($(MEXCC),)

ifneq ($(GPU),NO)
MEX_CUDA_SRCS+=$(patsubst %,%$(CU_EXT), $(VBLKR_SRC_NAMES))
MEX_GPU_FILES:=$(patsubst %,$(MEX_EXE_DIR)/%$(MEX_EXT), $(VBLKR_SRC_NAMES))
else
MEX_CPP_SRCS+=$(patsubst %,%.cc, $(VBLKR_SRC_NAMES))
MEX_CPU_FILES:=$(patsubst %,$(MEX_EXE_DIR)/%$(MEX_EXT), $(VBLKR_SRC_NAMES))
endif

endif				# ifneq ($(MEXCC),)

# TEST FILES
ifneq ($(GPU),NO)
TST_CUDA_EXE_FILES:=$(patsubst %$(CU_EXT),$(TST_EXE_DIR)/%$(EXE_EXT), $(G_TST_SRCS))
$(TST_CUDA_EXE_FILES):  $(TST_LIB_DIR)/libgputools$(LIB_EXT)
endif
TST_CPP_EXE_FILES:=$(patsubst %.cc,$(TST_EXE_DIR)/%$(EXE_EXT), $(C_TST_SRCS))
$(TST_CPP_EXE_FILES): $(TST_LIB_DIR)/libtools$(LIB_EXT) 

TST_EXE_FILES:=$(TST_CPP_EXE_FILES) $(TST_CUDA_EXE_FILES)

ifneq ($(MEXCC),)

MEX_CPP_EXE_FILES:=$(MEX_CPU_FILES)
MEX_CUDA_EXE_FILES:=$(MEX_GPU_FILES)
MEX_EXE_FILES:=$(MEX_CPU_FILES) $(MEX_GPU_FILES)

$(MEX_EXE_FILES): $(MEX_LIB_DIR)/libtools$(MEX_L_EXT)

endif				# ifneq ($(MEXCC),)

include ../util/rules.mak
