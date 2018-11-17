all : yuvb420pcs md_v3 md_v4

CS_MISC=../../cs_misc
CS_WHM_ENCODE=../../cs_whm_encode
RNDSTRMC=../../../../RndStrmC

# include path
INCLUDE_DIRS:=. $(CS_MISC) $(CS_WHM_ENCODE) $(RNDSTRMC) $(RNDSTRMC)/cgen/lib
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

TST_EXE_DIR:=$(TST_EXE_ROOT_DIR)/acq_ana.file

TST_CUDA_SRCS:=yuvb420pcs.cu md_v4.cu md_v3.cu

RND_LIB:=$(TST_LIB_DIR)/libcgenrnd.a
CS_WHM_ENC_LIB:=$(TST_LIB_DIR)/libcs_whmenc$(LIB_EXT)
CS_MISC_LIB:=$(TST_LIB_DIR)/libcs_misc$(LIB_EXT)

TST_CUDA_EXE_FILES:=$(patsubst %.cu,$(TST_EXE_DIR)/%$(EXE_EXT), $(TST_CUDA_SRCS))

$(TST_CUDA_EXE_FILES): $(CS_WHM_ENC_LIB) $(CS_MISC_LIB) $(RND_LIB)

TST_CUDA_SYS_LIBS:= -lm -lgomp /usr/local/lib/libjson-c.so $(OPENCV_LIBS) $(TST_CUDA_SYS_LIBS)

include $(ROOT_DIR)/util/rules.mak

# yuvb420pcs : yuvb420pcs.cu $(CS_WHM_ENC_LIB)  $(CS_MISC_LIB) $(RND_LIB)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT) \
#                 $(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		$(OPENCV_LIBS) \
# 		-ljson-c -lm -lgomp -o yuvb420pcs
# 
# md_v4 : md_v4.cu $(CS_WHM_ENC_LIB) $(CS_MISC_LIB) $(RND_LIB)
# 	nvcc -arch=compute_35 --compiler-bindir /usr/bin/gcc $(INCLUDE_OPT) \
#                 $(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		$(OPENCV_LIBS) \
# 		-ljson-c -lm -lgomp -o md_v4
# 
# md_v3 : md_v3.cu  $(CS_WHM_ENC_LIB) $(CS_MISC_LIB) $(RND_LIB)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT) \
#                 $(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		$(OPENCV_LIBS) \
# 		-ljson-c -lm -lgomp -o md_v3
# 
# clean :
# 	rm -f yuvb420pcs
# 	rm -f md_v3
