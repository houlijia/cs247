// 08/20/15 ::: orignally jianwei.txt in mr.pc/jianwei/jianwei.txt

cd /home/ldl/mr/baotou_cs/cs/transport_code/readWriteTest

make all

cs2rtp

-send file // tag for the rtp files ... in -output dir or in the input file dir
		// -input
    udp // need -ip -port
	std // to standard ...

-isFullSpeed // 1: no delay ... delay is from the frame rate, etc
-useDrop // debugging purpose

-----------------------------------------------------------------

rtp2cs

-recv file // tag for the rtp files ... need -input for dir
	udp // need -ipfrom -portFrom
	
-send // file ... file
	// tcp ... need -ipTo - portTo ... connect to matlab originally.

-output // dir name
-drop // no, uniform, fromFile (droplist.txt) in current dir ...
-dropN // 


---------------------------------------------------------------

home/jianwel/yuv/*sh	// demo ... in videoScripts now ...
