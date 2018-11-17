all : i_recon_test i_server_test

i_recon_test : i_recon_test.c ../ibuf.o ../i_recon.o ../serial_wht3.o
	g++ -o i_recon_test i_recon_test.c -lm ../i_recon.o ../ibuf.o ../serial_wht3.o

i_server_test: i_server_test.c ../i_recon.o ../ibuf.o ../i_recon.o ../../../cs_misc/cs_image.o 
	g++ -I. \
		../ibuf.o \
		../i_recon.o \
		../serial_wht3.o \
		../../../cs_misc/cs_image.o \
		-L/usr/local/lib \
		-lopencv_calib3d \
		-lopencv_contrib \
		-lopencv_core \
		-lopencv_features2d \
		-lopencv_flann \
		-lopencv_gpu \
		-lopencv_highgui \
		-lopencv_imgproc \
		-lopencv_legacy \
		-lopencv_ml \
		-lopencv_nonfree \
		-lopencv_objdetect \
		-lopencv_photo \
		-lopencv_stitching \
		-lopencv_superres \
		-lopencv_ts \
		-lopencv_video \
		-lopencv_videostab \
		-lm -o i_server_test i_server_test.c
