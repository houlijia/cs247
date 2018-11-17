# include path
INCLUDE_DIRS:=. ../cs_misc 
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

TST_CUDA_SRCS:=cs_whm_transp.cu cs_whm_encode_b.cu cs_whm_encode.cu

CS_WHM_ENC_LIB_FILES:=$(patsubst %.cu,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(TST_CUDA_SRCS))
CS_WHM_ENC_LIB:=$(TST_LIB_DIR)/libcs_whmenc$(LIB_EXT)
$(CS_WHM_ENC_LIB): $(CS_WHM_ENC_LIB_FILES)

TST_LIB_FILES:=$(CS_WHM_ENC_LIB)

include $(ROOT_DIR)/util/rules.mak

# all : cs_whm_encode.o cs_whm_encode_b.o cs_whm_transp.o
# 
# NVCC_OPTS=-arch=compute_35 -dc
# 
# cs_whm_transp.o : cs_whm_transp.cu cs_whm_transp.h ../cs_misc/cs_dbg.h ../cs_misc/cs_helper.h
# 	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../cs_misc -c cs_whm_transp.cu
# 
# cs_whm_encode_b.o : cs_whm_encode_b.cu cs_whm_encode_b.h ../cs_misc/cs_dbg.h ../cs_misc/cs_helper.h
# 	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../cs_misc -c cs_whm_encode_b.cu
# 
# cs_whm_encode.o : cs_whm_encode.cu cs_whm_encode.h
# 	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I ../cs_misc -c cs_whm_encode.cu
# 
# clean:
# 	rm -f cs_whm_encode.o
# 	rm -f cs_whm_encode_b.o
# 	rm -f cs_whm_transp.o
