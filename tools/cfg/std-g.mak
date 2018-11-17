include $(CFG_DIR)/std_basic.mak
include $(CFG_DIR)/nvcc_basic.mak

CC_OPTS += -DHAS_GPU=1
CPP_OPTS += -DHAS_GPU=1
MEX_OPTS += -DHAS_GPU=1
CU_EXT:=.cu

ifeq ($(GPU),)

GPU:=YES

else
ifneq ($(GPU),YES)
$(error In this configuration, GPU cannot get any value other than YES)
endif
endif

NVCC_OPTS+= --optimize 2 --compiler-options -O2 -DNDEBUG=1

