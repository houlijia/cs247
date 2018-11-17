all : resize_test lo_test iclient iclient2 iserver_gpu iserver_gpu2

H_DIRS=../camera_demo
O_DIR=../camera_demo
OBJS= \
	$(O_DIR)/ibuf.o \
	$(O_DIR)/serial_wht3.o \
	../../cs_misc/cs_image.o \
	../../cs_misc/cs_helper.o

RESIZEOBJS= \
	Nimresize/all.o          Nimresize/fprintf.o               Nimresize/Nimresize_terminate.o \
	Nimresize/any.o          Nimresize/meshgrid.o              Nimresize/rdivide.o \
	Nimresize/conv2.o        Nimresize/Nimresize_emxAPI.o      Nimresize/rot90.o \
	Nimresize/fileManager.o  Nimresize/Nimresize_emxutil.o     Nimresize/rtGetInf.o \
	Nimresize/filter2.o      Nimresize/Nimresize_initialize.o  Nimresize/rtGetNaN.o \
	Nimresize/floor.o        Nimresize/Nimresize.o             Nimresize/rt_nonfinite.o

TCP_SOCKETS= ./tcp_socket/AcceptTCPConnection.o ./tcp_socket/CreateTCPServerSocket.o \
	./tcp_socket/DieWithError.o ./tcp_socket/HandleTCPClient.o

file_io.o : file_io.c
	g++ -c -I. -INimresize file_io.c

main2.o : main2.c
	g++ -c -I. -INimresize main2.c

%.o : %.c
	g++ -I. -I$(H_DIRS) -c $<

iserver_gpu: iserver_gpu.cu
	make -C tcp_socket
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. \
		-DPLAN_B \
		$(TCP_SOCKETS) \
		-L/usr/local/lib \
		-lm -lgomp -o iserver_gpu iserver_gpu.cu

iserver_gpu2: iserver_gpu.cu
	make -C tcp_socket
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. \
		file_io.o \
		$(TCP_SOCKETS) \
		-L/usr/local/lib \
		-lm -lgomp -o iserver_gpu2 iserver_gpu.cu

iclient2: iclient2.cu  main2.o file_io.o localized_Ordered_sensing.o
	make -C Nimresize
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. \
	main2.o \
	$(RESIZEOBJS) \
	file_io.o \
	localized_Ordered_sensing.o \
	-lopencv_calib3d \
	-lopencv_core \
	-lopencv_features2d \
	-lopencv_flann \
	-lopencv_highgui \
	-lopencv_imgproc \
	-lopencv_ml \
	-lopencv_objdetect \
	-lopencv_photo \
	-lopencv_stitching \
	-lopencv_superres \
	-lopencv_ts \
	-lopencv_imgcodecs \
	-lopencv_video \
	-lopencv_videostab \
	-lgomp \
	-o iclient2 iclient2.cu 

iclient: iclient.cu do_resize.o
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. \
	$(OBJS) \
	do_resize.o \
	localized_Ordered_sensing.o \
	-lopencv_calib3d \
	-lopencv_core \
	-lopencv_features2d \
	-lopencv_flann \
	-lopencv_highgui \
	-lopencv_imgproc \
	-lopencv_ml \
	-lopencv_objdetect \
	-lopencv_photo \
	-lopencv_imgcodecs \
	-lopencv_stitching \
	-lopencv_superres \
	-lopencv_ts \
	-lopencv_video \
	-lopencv_videostab \
	-lgomp \
	-o iclient iclient.cu 

resize_test : resize_test.c do_resize.o do_bicubic.o
	g++ -I. \
		-I$(H_DIRS) \
		do_resize.o \
		do_bicubic.o \
		-o resize_test resize_test.c

lo_test : lo_test.cu localized_Ordered_sensing.o
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. \
		-I$(H_DIRS) \
		localized_Ordered_sensing.o \
		../../cs_misc/cs_helper.o \
		-lm \
		-lgomp \
		-o lo_test lo_test.cu

clean:
	rm -f *.o
	rm -f lo_test
	rm -f iclient
	rm -f resize_test iclient2 iserver_gpu iserver_gpu2
