all : yuvb420pcs

CS_MISC=../../cs_misc
CS_WHM_ENCODE=../../cs_whm_encode
RNDSTRMC=../../..

yuvb420pcs : yuvb420pcs.cu $(CS_MISC)/cs_interpolate.o $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o \
	$(CS_MISC)/cs_header.o $(CS_WHM_ENCODE)/cs_whm_encode_b.o $(CS_MISC)/cs_block.o \
	$(CS_MISC)/cs_copy_box.o $(CS_MISC)/cs_edge_detect_v2.o $(CS_MISC)/cs_motion_detect_v2.o \
	$(CS_MISC)/cs_perm_mlseq.o $(CS_MISC)/cs_expand.o $(CS_MISC)/cs_perm_selection.o \
	$(CS_MISC)/cs_config.o $(CS_MISC)/cs_motion_report.o $(CS_MISC)/cs_dct.o $(CS_MISC)/cs_ipcam.o  \
	$(CS_MISC)/cs_webcam.o 

	nvcc --compiler-bindir $(CUDA_CC) -I$(CS_WHM_ENCODE) -I$(CS_MISC) -I$(RNDSTRMC)/RndStrmC \
		-I$(RNDSTRMC)/RndStrmC/cgen/lib yuvb420pcs.cu \
		$(CS_MISC)/cs_dbg.o \
		$(CS_MISC)/cs_helper.o \
		$(CS_MISC)/cs_header.o \
		$(CS_MISC)/cs_block.o \
		$(CS_MISC)/cs_perm_mlseq.o \
		$(CS_MISC)/cs_expand.o \
		$(CS_MISC)/cs_interpolate.o \
		$(CS_MISC)/cs_perm_selection.o \
		$(CS_MISC)/cs_copy_box.o \
		$(CS_MISC)/cs_edge_detect_v2.o \
		$(CS_MISC)/cs_motion_detect_v2.o \
		$(CS_MISC)/cs_motion_report.o \
		$(CS_MISC)/cs_config.o \
		$(CS_MISC)/cs_ipcam.o \
		$(CS_MISC)/cs_webcam.o \
		$(CS_MISC)/cs_dct.o \
		$(CS_WHM_ENCODE)/cs_whm_encode_b.o \
		$(RNDSTRMC)/RndStrmC/libcsgenlib.a \
		/usr/local/lib/libopencv_calib3d.so \
		/usr/local/lib/libopencv_contrib.so \
		/usr/local/lib/libopencv_core.so \
		/usr/local/lib/libopencv_features2d.so \
		/usr/local/lib/libopencv_flann.so \
		/usr/local/lib/libopencv_gpu.so \
		/usr/local/lib/libopencv_highgui.so \
		/usr/local/lib/libopencv_imgproc.so \
		/usr/local/lib/libopencv_legacy.so \
		/usr/local/lib/libopencv_ml.so \
		/usr/local/lib/libopencv_nonfree.so \
		/usr/local/lib/libopencv_objdetect.so \
		/usr/local/lib/libopencv_photo.so \
		/usr/local/lib/libopencv_stitching.so \
		/usr/local/lib/libopencv_superres.so \
		/usr/local/lib/libopencv_ts.so \
		/usr/local/lib/libopencv_video.so \
		/usr/local/lib/libopencv_videostab.so \
		-ljson-c -lm -lgomp -o yuvb420pcs

clean :
	rm -f yuvb420pcs
