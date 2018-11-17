# include path
INCLUDE_DIRS:=. ../cuda_tst
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include ../util/defs.mak

# TCP communication, only in C, CPU only
COMMON_TCP_TST_SRCS:= tcp_sock_io.c ip_util.c ip_name.c timeval.c

ifneq ($(MEXCC),)

COMMON_TCP_MEX_SRCS:=$(COMMON_TCP_TST_SRCS) tcp_sock_mex.c

# Sources which correspond to executable, i.e. contain mexFunction().
MEX_TCP_SRCS:=$(patsubst %,%_mex.c,openTCPSocketServer openTCPSocketClient \
  closeTCPSocket sendTCPSocket recvTCPSocket undefSocket)
MEX_C_SRCS:=$(MEX_TCP_SRCS) $(COMMON_TCP_MEX_SRCS)

MEX_TCP_FILES:=$(patsubst %.c, $(MEX_EXE_DIR)/%$(MEX_EXT), $(MEX_TCP_SRCS))

$(MEX_TCP_FILES): $(patsubst %.c, $(MEX_OBJ_DIR)/%$(MEX_O_EXT), $(COMMON_TCP_MEX_SRCS))

endif				# ifneq ($(MEXCC),)

TST_TCP_SRCS:= tcp_sock_test.c
TST_C_SRCS:= $(TST_TCP_SRCS) $(COMMON_TCP_TST_SRCS)

TST_TCP_FILES:=$(patsubst %.c, $(TST_EXE_DIR)/%$(EXE_EXT), $(TST_TCP_SRCS))

$(TST_TCP_FILES): $(patsubst %.c, $(TST_OBJ_DIR)/%$(OBJ_EXT), $(COMMON_TCP_TST_SRCS))

# Coding, C++, GPU or CPU

ifneq ($(GPU),NO)
TST_GPU_CODER_SRCS:=cuda_CodeDest_tst$(CU_EXT)
TST_GPU_CODER_COMMON_SRCS:=cuda_CodeDest_tst_do$(CU_EXT) cuda_CodeDest$(CU_EXT)
TST_GPU_CODER_FILES:=$(patsubst %$(CU_EXT),$(TST_EXE_DIR)/%$(EXE_EXT), $(TST_GPU_CODER_SRCS))
$(TST_GPU_CODER_FILES): $(patsubst %.cc,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(TST_GPU_CODER_COMMON_SRCS))

ifneq ($(MEXCC),)

MEX_GPU_CODER_SRCS:=encodeUInt_mex$(CU_EXT) encodeSInt_mex$(CU_EXT) encodeNumber_mex$(CU_EXT)
MEX_GPU_CODER_COMMON_SRCS:= cuda_CodeDest$(CU_EXT)

MEX_GPU_CODER_FILES:=$(patsubst %$(CU_EXT),$(MEX_EXE_DIR)/%$(MEX_EXT), $(MEX_GPU_CODER_SRCS))
$(MEX_GPU_CODER_FILES): $(patsubst %.$(CU_EXT),$(MEX_OBJ_DIR)/%$(MEX_O_EXT), $(MEX_GPU_CODER_COMMON_SRCS))
$(TST_GPU_CODER_FILES): $(TST_LIB_DIR)/libgputools$(LIB_EXT)

endif				# ifneq ($(MEXCC),)

else

ifneq ($(MEXCC),)
MEX_CPU_CODER_SRCS:= deleteCodeElement_mex.cc constructCodeElement_mex.cc 

MEX_CPU_CODER_SRCS+= encodeUInt_mex.cc encodeSInt_mex.cc encodeNumber_mex.cc
MEX_CPU_CODER_COMMON_SRCS:=
endif

endif

TST_CPU_CODER_SRCS:= cc_CodeDest_tst.cc
TST_CPU_CODER_COMMON_SRCS:=cc_CodeDest_tst_do.cc cc_CodeDest.cc
TST_CPU_CODER_FILES:=$(patsubst %.cc,$(TST_EXE_DIR)/%$(EXE_EXT), $(TST_CPU_CODER_SRCS))
$(TST_CPU_CODER_FILES): $(patsubst %.cc,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(TST_CPU_CODER_COMMON_SRCS))
$(TST_CPU_CODER_FILES): $(TST_LIB_DIR)/libtools$(LIB_EXT)


TST_CPP_SRCS:=$(TST_CPU_CODER_SRCS) $(TST_CPU_CODER_COMMON_SRCS)
TST_CUDA_SRCS:=$(TST_GPU_CODER_SRCS) $(TST_GPU_CODER_COMMON_SRCS)

TST_C_EXE_FILES:=$(TST_TCP_FILES)
TST_CPP_EXE_FILES:=$(TST_CPU_CODER_FILES)
TST_CUDA_EXE_FILES:=$(TST_GPU_CODER_FILES)

TST_CC_SYS_LIBS:=-lm

ifneq ($(MEXCC),)

MEX_CPU_CODER_FILES:=$(patsubst %.cc,$(MEX_EXE_DIR)/%$(MEX_EXT), $(MEX_CPU_CODER_SRCS))
$(MEX_CPU_CODER_FILES): $(patsubst %.cc,$(MEX_OBJ_DIR)/%$(MEX_O_EXT), $(MEX_CPU_CODER_COMMON_SRCS))
$(MEX_CPU_CODER_FILES) $(MEX_GPU_CODER_FILES): $(MEX_LIB_DIR)/libtools$(MEX_L_EXT)

MEX_CPP_SRCS:=$(MEX_CPU_CODER_SRCS) $(MEX_CPU_CODER_COMMON_SRCS)
MEX_CUDA_SRCS:=$(MEX_GPU_CODER_SRCS) $(MEX_GPU_CODER_COMMON_SRCS)

MEX_C_EXE_FILES:=$(MEX_TCP_FILES)
MEX_CPP_EXE_FILES:=$(MEX_CPU_CODER_FILES)
MEX_CUDA_EXE_FILES:=$(MEX_GPU_CODER_FILES)

ifeq ($(DO_M_DPND),YES)

MEX_LIBS+= 
ifeq ($(MEX_COMPILER),MSVC)	# Windows MS Visual C case
MEX_LIBS+= -lwinmm -lWs2_32
endif

ifeq ($(MEX_COMPILER),MinGW)	# MinGW case
ifeq ($(origin WSOCKLIB),undefined)


WSOCKLIBDIR:= $(shell \
  find $(MATLAB_PATH)/../SupportPackages -type d -name 'MW_MinGW*' -prune -exec find '{}' -type d -name lib -prune \; )
export WSOCKLIB:=$(wildcard $(patsubst %,%/libws2_32.a, $(WSOCKLIBDIR)))
MEX_LIBS+= $(WSOCKLIB)
$(info Added to MEX_LIBS: $(WSOCKLIB))

endif
endif

endif

endif				# ifneq ($(MEXCC),)

include ../util/rules.mak


