NVCC_OPTS:= $(INCLUDE_OPT) -arch=compute_35 --compiler-bindir $(CUDA_CC) \
  --compiler-options -Werror,-Wall,-ansi,-O0,-g -DHAS_GPU=1

ifneq ($(filter kanas%,$(HOSTNAME)),)
NVCC_OPTS+= -DOPENCV_3
endif
