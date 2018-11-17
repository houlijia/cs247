include $(CFG_DIR)/dbg_basic.mak

CC_OPTS += -DHAS_GPU=0
CPP_OPTS += -DHAS_GPU=0
MEX_OPTS += -DHAS_GPU=0

ifeq ($(GPU),)

GPU:=NO

else
ifneq ($(GPU),NO)
$(error In this configuration, GPU cannot get any value other than NO)
endif
endif

