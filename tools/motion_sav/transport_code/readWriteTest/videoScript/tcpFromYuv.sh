file1="10-all.o.yuv"
file3="point-15-all.o.yuv"
file5="20-all.o.yuv"
file2="ori-75k.yuv"
file4="ori-20k.yuv"
file6="ori-4k.yuv"
ffmpeg -s 352x288 -i $file1\
       -s 352x288  -i $file2\
       -s 352x288 -i $file3\
       -s 352x288  -i $file4\
       -s 352x288 -i $file5\
       -s 352x288  -i $file6\
  -t 00:00:39.50 \
  -r 30 -qscale:v 3\
	-filter_complex \
	"[0:v:0]pad=iw*3:ih*2[bg];\
	[bg][1:v:0]overlay=0:h[mid0];\
	[mid0][2:v:0]overlay=w[mid1];\
	[mid1][3:v:0]overlay=w:h[mid2];\
	[mid2][4:v:0]overlay=w*2:0[mid3];\
	[mid3][5:v:0]overlay=w*2:h"	tcpFromyuv.mp4

	
