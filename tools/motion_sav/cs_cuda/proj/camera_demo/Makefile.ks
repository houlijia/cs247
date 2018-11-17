all : iclient iserver iserver_gpu iserver_gpu_thread

TOOL_DIR=../../tool
TCP_DIR=./tcp_socket
MISC_DIR=../../cs_misc
WHM_DIR=../../cs_whm_encode

TCP_SOCKETS= ./tcp_socket/AcceptTCPConnection.o ./tcp_socket/CreateTCPServerSocket.o \
	./tcp_socket/DieWithError.o ./tcp_socket/HandleTCPClient.o

%.o : %.c
	g++ -I. -I$(TOOL_DIR) -c $<

iclient : iclient.cu serial_wht3.o tables.o ../../cs_misc/cs_helper.o
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. -I$(TOOL_DIR) \
		-g -lm -lgomp ../../cs_misc/cs_helper.o tables.o serial_wht3.o -o iclient iclient.cu

iserver: iserver.cu ibuf.o rgb2jpg.o serial_wht3.o i_recon.o tables.o \
		../../cs_misc/cs_helper.o ../../cs_misc/cs_image.o 
	make -C tcp_socket
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. \
		$(TCP_SOCKETS) \
		ibuf.o \
		i_recon.o \
		tables.o \
		serial_wht3.o \
		rgb2jpg.o \
		../../cs_misc/cs_image.o \
		../../cs_misc/cs_helper.o \
		-L/usr/local/lib \
		-L/usr/lib64 \
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
		-lopencv_imgcodecs \
		-lopencv_superres \
		-lopencv_ts \
		-lopencv_video \
		-lopencv_videostab \
		-ljpeg \
		-lm -lgomp -o iserver iserver.cu

iserver_gpu: iserver_gpu.cu ibuf.o rgb2jpg.o serial_wht3.o tables.o \
		$(MISC_DIR)/cs_recon.o \
		$(MISC_DIR)/cs_helper.o \
		$(MISC_DIR)/cs_image.o \
		$(MISC_DIR)/cs_buffer.o \
		$(MISC_DIR)/cs_matrix.o \
		$(MISC_DIR)/cs_perm_generic.o \
		$(MISC_DIR)/cs_dbg.o \
		$(WHM_DIR)/cs_whm_encode_b.o
	make -C tcp_socket
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. \
		$(TCP_SOCKETS) \
		ibuf.o \
		tables.o \
		serial_wht3.o \
		rgb2jpg.o \
		$(MISC_DIR)/cs_recon.o \
		$(MISC_DIR)/cs_dbg.o \
		$(MISC_DIR)/cs_image.o \
		$(MISC_DIR)/cs_helper.o \
		$(MISC_DIR)/cs_matrix.o \
		$(MISC_DIR)/cs_buffer.o \
		$(MISC_DIR)/cs_perm_generic.o \
		$(WHM_DIR)/cs_whm_encode_b.o \
		-L/usr/local/lib \
		-L/usr/lib64 \
		-lopencv_calib3d \
		-lopencv_core \
		-lopencv_features2d \
		-lopencv_flann \
		-lopencv_highgui \
		-lopencv_imgproc \
		-lopencv_imgcodecs \
		-lopencv_ml \
		-lopencv_objdetect \
		-lopencv_photo \
		-lopencv_stitching \
		-lopencv_superres \
		-lopencv_ts \
		-lopencv_video \
		-lopencv_videostab \
		-ljpeg \
		-lm -lgomp -o iserver_gpu iserver_gpu.cu

iserver_gpu_thread: iserver_gpu.cu ibuf.o rgb2jpg.o serial_wht3.o tables.o \
		$(MISC_DIR)/cs_recon_thread.o \
		$(MISC_DIR)/cs_helper.o \
		$(MISC_DIR)/cs_image.o \
		$(MISC_DIR)/cs_buffer.o \
		$(MISC_DIR)/cs_matrix.o \
		$(MISC_DIR)/cs_perm_generic.o \
		$(MISC_DIR)/cs_dbg.o \
		$(WHM_DIR)/cs_whm_encode_b.o
	make -C tcp_socket
	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I. \
		$(TCP_SOCKETS) \
		ibuf.o \
		tables.o \
		serial_wht3.o \
		rgb2jpg.o \
		$(MISC_DIR)/cs_recon_thread.o \
		$(MISC_DIR)/cs_dbg.o \
		$(MISC_DIR)/cs_image.o \
		$(MISC_DIR)/cs_helper.o \
		$(MISC_DIR)/cs_matrix.o \
		$(MISC_DIR)/cs_buffer.o \
		$(MISC_DIR)/cs_perm_generic.o \
		$(WHM_DIR)/cs_whm_encode_b.o \
		-L/usr/local/lib \
		-L/usr/lib64 \
		-lopencv_calib3d \
		-lopencv_core \
		-lopencv_features2d \
		-lopencv_flann \
		-lopencv_highgui \
		-lopencv_imgproc \
		-lopencv_imgcodecs \
		-lopencv_ml \
		-lopencv_objdetect \
		-lopencv_photo \
		-lopencv_stitching \
		-lopencv_superres \
		-lopencv_ts \
		-lopencv_video \
		-lopencv_videostab \
		-ljpeg \
		-lm -lgomp -o iserver_gpu_thread iserver_gpu.cu

clean :
	rm -f *.o
	rm -f iserver iserver_gpu
	rm -f iclient
