# all : resize_test lo_test iclient iclient2 iserver_gpu iserver_gpu2

CS_MISC=../../cs_misc
H_DIRS=../camera_demo
O_DIR=../camera_demo
CS_WHM_ENCODE=../../cs_whm_encode

# include path
INCLUDE_DIRS:=. Nimresize $(H_DIRS) $(CS_MISC)
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

TST_EXE_DIR:=$(TST_EXE_ROOT_DIR)/obj_recog

CUDA_EXE_SRCS:=iserver_gpu.cu iserver_gpu2.cu iclient2.cu iclient.cu lo_test.cu
CPP_EXE_SRCS:=resize_test.cc

TST_DIRS_USED+= $(TST_OBJ_DIR)/Nimresize
DIRS_USED+= $(TST_OBJ_DIR)/Nimresize

RESIZE_SRCS:=$(patsubst %,Nimresize/%.c, \
  all fprintf Nimresize_terminate any meshgrid rdivide conv2 Nimresize_emxAPI rot90 \
  fileManager Nimresize_emxutil rtGetInf filter2 Nimresize_initialize rtGetNaN \
  floor Nimresize rt_nonfinite)

TST_CUDA_SRCS:=$(CUDA_EXE_SRCS)
TST_CPP_SRCS:=do_bicubic.cc  do_resize.cc  file_io.cc  localized_Ordered_sensing.cc \
   main2.cc  resize_test.cc
TST_C_SRCS:=$(RESIZE_SRCS)

RESIZE_OBJS:=$(patsubst %.c,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(RESIZE_SRCS))

CS_WHM_ENC_LIB:=$(TST_LIB_DIR)/libcs_whmenc$(LIB_EXT)
RND_LIB:=$(TST_LIB_DIR)/libcgenrnd.a
CS_MISC_LIB:=$(TST_LIB_DIR)/libcs_misc$(LIB_EXT)
CS_CAMERA_DEMO_LIB:=$(TST_LIB_DIR)/libcs_camera_demo$(LIB_EXT)
COMMON_LIBS:=$(CS_WHM_ENC_LIB) $(CS_MISC_LIB) $(RND_LIB) $(CS_CAMERA_DEMO_LIB)

CS_TCP_SOCKET_LIB:= $(TST_LIB_DIR)/libcs_tcp_socket$(LIB_EXT) 

TST_CUDA_EXE_FILES:=$(patsubst %.cu,$(TST_EXE_DIR)/%$(EXE_EXT), $(CUDA_EXE_SRCS))
TST_CPP_EXE_FILES:= $(patsubst %.cc,$(TST_EXE_DIR)/%$(EXE_EXT), $(CPP_EXE_SRCS))

iserver_gpu_OPTS:= -DPLAN_B
$(patsubst %,$(TST_EXE_DIR)/%$(EXE_EXT), iserver_gpu iserver_gpu2): \
  $(TST_OBJ_DIR)/file_io$(OBJ_EXT) $(CS_TCP_SOCKET_LIB)

$(TST_EXE_DIR)/iclient2$(EXE_EXT): \
  $(patsubst %,$(TST_OBJ_DIR)/%$(OBJ_EXT), main2 file_io localized_Ordered_sensing) \
  $(RESIZE_OBJS)

$(TST_EXE_DIR)/iclient$(EXE_EXT): \
  $(patsubst %,$(TST_OBJ_DIR)/%$(OBJ_EXT), do_resize localized_Ordered_sensing) \
  $(COMMON_LIBS)

$(TST_EXE_DIR)/resize_test$(EXE_EXT): \
  $(patsubst %,$(TST_OBJ_DIR)/%$(OBJ_EXT), do_resize do_bicubic)

$(TST_EXE_DIR)/lo_test$(EXE_EXT): \
  $(patsubst %,$(TST_OBJ_DIR)/%$(OBJ_EXT), localized_Ordered_sensing) \
  $(COMMON_LIBS)

TST_CUDA_SYS_LIBS:= -lm -lgomp -ljpeg /usr/local/lib/libjson-c.so $(OPENCV_LIBS) $(TST_CUDA_SYS_LIBS)

include $(ROOT_DIR)/util/rules.mak



# 
# RESIZEOBJS= \
# 	Nimresize/all.o          Nimresize/fprintf.o               Nimresize/Nimresize_terminate.o \
# 	Nimresize/any.o          Nimresize/meshgrid.o              Nimresize/rdivide.o \
# 	Nimresize/conv2.o        Nimresize/Nimresize_emxAPI.o      Nimresize/rot90.o \
# 	Nimresize/fileManager.o  Nimresize/Nimresize_emxutil.o     Nimresize/rtGetInf.o \
# 	Nimresize/filter2.o      Nimresize/Nimresize_initialize.o  Nimresize/rtGetNaN.o \
# 	Nimresize/floor.o        Nimresize/Nimresize.o             Nimresize/rt_nonfinite.o
# 
# file_io.o : file_io.c
# 	g++ -c -I. -INimresize file_io.c
# 
# main2.o : main2.c
# 	g++ -c -I. -INimresize main2.c
# 
# %.o : %.c
# 	g++ -I. -I$(H_DIRS) -c $<
# 
# iserver_gpu: iserver_gpu.cu file_io.o $(CS_TCP_SOCKET_LIB)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. \
# 		-DPLAN_B \
# 		$(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		-L/usr/local/lib \
# 		-lm -lgomp -o iserver_gpu
# 
# iserver_gpu2: iserver_gpu.cu file_io.o $(CS_TCP_SOCKET_LIB)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. \
# 		$(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		-L/usr/local/lib \
# 		-lm -lgomp -o iserver_gpu2
# 
# iclient2: iclient2.cu  main2.o file_io.o localized_Ordered_sensing.o $(RESIZEOBJS)
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. \
# 	$(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
#         $(OPENCV_LIBS) \
# 	-o iclient2
# 
# iclient: iclient.cu do_resize.o localized_Ordered_sensing.o $(COMMON_LIBS) 
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. \
# 	$(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
#        $(OPENCV_LIBS) \
# 	-lgomp -ljson-c \
# 	-o iclient
# 
# resize_test : resize_test.c do_resize.o do_bicubic.o
# 	g++ -I. -I$(H_DIRS) \
# 	$(filter %.c %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
# 		-o resize_test
# 
# lo_test : lo_test.cu localized_Ordered_sensing.o $(COMMON_LIBS) 
# 	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) -I. -I$(H_DIRS) \
# 		$(filter %.cu %$(OBJ_EXT) %$(LIB_EXT), $^) \
#                 $(OPENCV_LIBS) \
# 		-lm -lgomp -ljson-c \
# 		-o lo_test
# 
# clean:
# 	rm -f *.o
# 	rm -f lo_test
# 	rm -f iclient resize_test iclient2 iserver_gpu iserver_gpu2
# 
