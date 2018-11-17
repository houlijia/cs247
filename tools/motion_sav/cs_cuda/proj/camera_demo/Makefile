# all : iclient iserver iserver_gpu iserver_gpu_thread

TOOL_DIR=../../tool
TCP_DIR=./tcp_socket
MISC_DIR=../../cs_misc
WHM_DIR=../../cs_whm_encode

# include path
INCLUDE_DIRS:=. ../../tool $(CS_MISC) $(CS_WHM_ENCODE) $(RNDSTRMC) $(RNDSTRMC)/cgen/lib
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

TST_EXE_DIR:=$(TST_EXE_ROOT_DIR)/camera_demo

SERVER_CUDA_SRCS:=iserver.cu iserver_gpu.cu iserver_gpu_thread.cu
TST_CUDA_SRCS:=iclient.cu $(SERVER_CUDA_SRCS)
TST_CPP_SRCS:=ibuf.cc i_recon.cc rgb2jpg.cc serial_wht3.cc
TST_C_SRCS:=tables.c 

CS_CAMERA_DEMO_LIB:=$(TST_LIB_DIR)/libcs_camera_demo$(LIB_EXT)
$(CS_CAMERA_DEMO_LIB): $(patsubst %,$(TST_OBJ_DIR)/%$(OBJ_EXT), ibuf serial_wht3)
TST_LIB_FILES:=$(CS_CAMERA_DEMO_LIB)

RND_LIB:=$(TST_LIB_DIR)/libcgenrnd.a
CS_WHM_ENC_LIB:=$(TST_LIB_DIR)/libcs_whmenc$(LIB_EXT)
CS_MISC_LIB:=$(TST_LIB_DIR)/libcs_misc$(LIB_EXT)
CS_TCP_SOCKET_LIB:= $(TST_LIB_DIR)/libcs_tcp_socket$(LIB_EXT)
COMMON_LIBS:=$(CS_WHM_ENC_LIB) $(CS_MISC_LIB) $(RND_LIB) $(CS_TCP_SOCKET_LIB) $(CS_CAMERA_DEMO_LIB)

CS_RECON_NO_THRD_LIB:=$(TST_LIB_DIR)/librecon_nothrd$(LIB_EXT)
CS_RECON_THRD_LIB:=$(TST_LIB_DIR)/librecon_thrd$(LIB_EXT)

TST_CUDA_EXE_FILES:=$(patsubst %.cu,$(TST_EXE_DIR)/%$(EXE_EXT), $(TST_CUDA_SRCS))

$(TST_CUDA_EXE_FILES): $(TST_OBJ_DIR)/tables$(OBJ_EXT) $(COMMON_LIBS)

$(patsubst %.cu,$(TST_EXE_DIR)/%$(EXE_EXT), $(SERVER_CUDA_SRCS)): \
  $(TST_OBJ_DIR)/rgb2jpg$(OBJ_EXT)

$(TST_EXE_DIR)/iserver$(EXE_EXT): $(TST_OBJ_DIR)/i_recon$(OBJ_EXT)
$(TST_EXE_DIR)/iserver_gpu$(EXE_EXT): $(CS_RECON_NO_THRD_LIB)
$(TST_EXE_DIR)/iserver_gpu_thread$(EXE_EXT): $(CS_RECON_THRD_LIB)

TST_CUDA_SYS_LIBS:= -lm -lgomp -ljpeg /usr/local/lib/libjson-c.so $(OPENCV_LIBS) $(TST_CUDA_SYS_LIBS)

include $(ROOT_DIR)/util/rules.mak


# 
# %.o : %.c
# 	g++ -I. -I$(TOOL_DIR) -c $<
# 
# iclient : iclient.cu serial_wht3.o tables.o $(COMMON_LIBS)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. -I$(TOOL_DIR) \
# 		-g -lm -lgomp  -o iclient \
# 		$(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		$(RND_LIB) $(OPENCV_LIBS) /usr/local/lib/libjson-c.so
# 
# iserver: iserver.cu ibuf.o rgb2jpg.o serial_wht3.o i_recon.o tables.o $(COMMON_LIBS)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. \
#                 $(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		$(OPENCV_LIBS) /usr/local/lib/libjson-c.so \
# 		-ljpeg \
# 		-lm -lgomp -o iserver
# 
# iserver_gpu: iserver_gpu.cu ibuf.o rgb2jpg.o serial_wht3.o tables.o \
#                 $(COMMON_LIBS) $(CS_RECON_NO_THRD_LIB)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. \
#                 $(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		$(OPENCV_LIBS) /usr/local/lib/libjson-c.so \
# 		-ljpeg \
# 		-lm -lgomp -o iserver_gpu
# 
# iserver_gpu_thread: iserver_gpu.cu ibuf.o rgb2jpg.o serial_wht3.o tables.o \
#                  $(COMMON_LIBS) $(CS_RECON_THRD_LIB)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. \
#                 $(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		$(OPENCV_LIBS) /usr/local/lib/libjson-c.so \
# 		-ljpeg \
# 		-lm -lgomp -o iserver_gpu_thread
# 
# clean :
# 	rm -f *.o
# 	rm -f iserver iserver_gpu
# 	rm -f iclient
# 
