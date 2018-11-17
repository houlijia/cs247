DBG=NO

# Determine if there is a GPU
ifeq ($(GPU),) # GPU not defined case

ifeq ($(strip $(shell which nvcc 2>/dev/null)),)

$(info nvcc not found. Assuming GPU=NO)
GPU:=NO

else

$(info nvcc found. Assuming GPU=YES)
GPU:=YES

endif

else  	# GPU is defined

ifneq ($(GPU),NO)
ifneq ($(GPU),YES)
$(warning GPU is $(GPU). Assuming it is yes)
override GPU=YES
endif
endif

endif

ifeq ($(GPU),NO)
include $(CFG_DIR)/std-ng.mak
else
include $(CFG_DIR)/std-g.mak
endif

