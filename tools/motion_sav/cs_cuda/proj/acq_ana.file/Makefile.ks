all : yuvb420pcs md_v3 md_v4

CS_MISC=../../cs_misc
CS_WHM_ENCODE=../../cs_whm_encode
RNDSTRMC=../../..

yuvb420pcs : yuvb420pcs.cu $(CS_MISC)/cs_interpolate.o $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o \
	$(CS_MISC)/cs_header.o $(CS_WHM_ENCODE)/cs_whm_encode_b.o $(CS_MISC)/cs_block.o \
	$(CS_MISC)/cs_copy_box.o $(CS_MISC)/cs_edge_detect_v2.o $(CS_MISC)/cs_motion_detect_v2.o \
	$(CS_MISC)/cs_perm_mlseq.o $(CS_MISC)/cs_expand.o $(CS_MISC)/cs_perm_selection.o \
	$(CS_MISC)/cs_config.o $(CS_MISC)/cs_motion_report.o $(CS_MISC)/cs_dct.o $(CS_MISC)/cs_video_io.o  \
	$(CS_MISC)/cs_webcam.o 

	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I$(CS_WHM_ENCODE) -I$(CS_MISC) -I$(RNDSTRMC)/RndStrmC \
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
		$(CS_MISC)/cs_video_io.o \
		$(CS_MISC)/cs_webcam.o \
		$(CS_MISC)/cs_dct.o \
		$(CS_WHM_ENCODE)/cs_whm_encode_b.o \
		$(RNDSTRMC)/RndStrmC/libcsgenlib.a \
		-L/usr/local/lib \
		-L/usr/lib \
		-lopencv_calib3d \
		-lopencv_core \
		-lopencv_features2d \
		-lopencv_flann \
		-lopencv_highgui \
		-lopencv_imgproc \
		-lopencv_videoio \
		-lopencv_ml \
		-lopencv_objdetect \
		-lopencv_photo \
		-lopencv_stitching \
		-lopencv_superres \
		-lopencv_ts \
		-lopencv_video \
		-lopencv_videostab \
		-ljson-c -lm -lgomp -o yuvb420pcs

md_v4 : md_v4.cu $(CS_MISC)/cs_interpolate.o $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o \
	$(CS_MISC)/cs_header.o $(CS_WHM_ENCODE)/cs_whm_encode_b.o $(CS_MISC)/cs_block.o \
	$(CS_MISC)/cs_copy_box.o $(CS_MISC)/cs_edge_detect_v2.o $(CS_MISC)/cs_motion_detect_v2.o \
	$(CS_MISC)/cs_perm_mlseq.o $(CS_MISC)/cs_expand.o $(CS_MISC)/cs_perm_selection.o \
	$(CS_MISC)/cs_config.o $(CS_MISC)/cs_motion_report.o $(CS_MISC)/cs_dct.o $(CS_MISC)/cs_video_io.o  \
	$(CS_MISC)/cs_webcam.o $(CS_MISC)/cs_motion_detect_v4.o $(CS_MISC)/cs_mean_sd.o \
	$(CS_MISC)/cs_quantize.o

	nvcc -arch=sm_20 --compiler-bindir /usr/bin/gcc -I../../cs_whm_encode \
		-I../../cs_misc -I../../../RndStrmC \
		-I../../../RndStrmC/cgen/lib md_v4.cu \
		../../cs_misc/cs_dbg.o \
		../../cs_misc/cs_helper.o \
		../../cs_misc/cs_header.o \
		../../cs_misc/cs_block.o \
		../../cs_misc/cs_perm_mlseq.o \
		../../cs_misc/cs_expand.o \
		../../cs_misc/cs_interpolate.o \
		../../cs_misc/cs_perm_selection.o \
		../../cs_misc/cs_copy_box.o \
		../../cs_misc/cs_edge_detect_v2.o \
		../../cs_misc/cs_motion_detect_v2.o \
		../../cs_misc/cs_motion_detect_v4.o \
		../../cs_misc/cs_motion_report.o \
		../../cs_misc/cs_config.o \
		../../cs_misc/cs_video_io.o \
		../../cs_misc/cs_webcam.o \
		../../cs_misc/cs_dct.o \
		../../cs_misc/cs_mean_sd.o \
		../../cs_misc/cs_quantize.o \
		../../cs_whm_encode/cs_whm_encode_b.o \
		../../../RndStrmC/libcsgenlib.a \
		-L/usr/local/lib \
		-L/usr/lib \
		-lopencv_calib3d \
		-lopencv_core \
		-lopencv_features2d \
		-lopencv_flann \
		-lopencv_highgui \
		-lopencv_imgproc \
		-lopencv_ml \
		-lopencv_objdetect \
		-lopencv_videoio \
		-lopencv_photo \
		-lopencv_stitching \
		-lopencv_superres \
		-lopencv_ts \
		-lopencv_video \
		-lopencv_videostab \
		-ljson-c -lm -lgomp -o md_v4

md_v3 : md_v3.cu $(CS_MISC)/cs_interpolate.o $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o \
	$(CS_MISC)/cs_header.o $(CS_WHM_ENCODE)/cs_whm_encode_b.o $(CS_MISC)/cs_block.o \
	$(CS_MISC)/cs_copy_box.o $(CS_MISC)/cs_edge_detect_v2.o $(CS_MISC)/cs_motion_detect_v3.o \
	$(CS_MISC)/cs_perm_mlseq.o $(CS_MISC)/cs_expand.o $(CS_MISC)/cs_perm_selection.o \
	$(CS_MISC)/cs_config.o $(CS_MISC)/cs_motion_report.o $(CS_MISC)/cs_dct.o $(CS_MISC)/cs_video_io.o  \
	$(CS_MISC)/cs_webcam.o 

	nvcc -arch=sm_20 --compiler-bindir $(CUDA_CC) -I$(CS_WHM_ENCODE) -I$(CS_MISC) -I$(RNDSTRMC)/RndStrmC \
		-I$(RNDSTRMC)/RndStrmC/cgen/lib md_v3.cu \
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
		$(CS_MISC)/cs_motion_detect_v3.o \
		$(CS_MISC)/cs_motion_report.o \
		$(CS_MISC)/cs_config.o \
		$(CS_MISC)/cs_video_io.o \
		$(CS_MISC)/cs_webcam.o \
		$(CS_MISC)/cs_dct.o \
		$(CS_WHM_ENCODE)/cs_whm_encode_b.o \
		$(RNDSTRMC)/RndStrmC/libcsgenlib.a \
		-L/usr/local/bin \
		-L/usr/lib \
		-lopencv_calib3d \
		-lopencv_core \
		-lopencv_features2d \
		-lopencv_videoio \
		-lopencv_flann \
		-lopencv_highgui \
		-lopencv_imgproc \
		-lopencv_ml \
		-lopencv_objdetect \
		-lopencv_photo \
		-lopencv_stitching \
		-lopencv_superres \
		-lopencv_ts \
		-lopencv_video \
		-lopencv_videostab \
		-ljson-c -lm -lgomp -o md_v3

clean :
	rm -f yuvb420pcs
	rm -f md_v3
