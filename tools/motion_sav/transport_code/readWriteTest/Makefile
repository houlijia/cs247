CC=gcc -g -O0 
CXX=g++ -g -O0


rtp2cs: rtp2cs.cpp RTP2CSFactory.h CircularQueue.h  readUInt.o common_ljw.o FileBufferSender.h writeUInt.o RtpPacket.o CodeElement.o udp_client_server.o tcpconnect.o 
        
cs2rtp: cs2rtp.cpp common_ljw.o writeUInt.o readUInt.o CodeElement.o TypeList.o readSInt.o readRealNumber.o UniformQuantizer.o RtpPacket.o UdpRTPSender.o udp_client_server.o
        

tcpconnect.o: tcpconnect.c tcpconnect.h

        

        
UdpRTPSender.o: UdpRTPSender.h UdpRTPSender.cpp udp_client_server.o
        
udp_client_server.o: udp_client_server.h udp_client_server.cpp
        
#testReadSInt: testReadSInt.cpp readSInt.o

#FileBufferSender.o: FileBufferSender.h FileBufferSender.cpp


#CircularQueue.o:  CircularQueue.cpp CircularQueue.h

no_limit_cs2rtp: no_limit_cs2rtp.cpp common_ljw.o writeUInt.o readUInt.o CodeElement.o TypeList.o readSInt.o readRealNumber.o UniformQuantizer.o RtpPacket.o

no_limit_rtp2cs: no_limit_rtp2cs.cpp common_ljw.o  RtpPacket.o

all: readFile cs2rtp rtp2cs no_limit_rtp2cs no_limit_cs2rtp 

readFile: readFile.cpp common_ljw.o writeUInt.o readUInt.o CodeElement.o TypeList.o readSInt.o readRealNumber.o UniformQuantizer.o

#testRtpPacket: testRtpPacket.cpp RtpPacket.o

RtpPacket.o: RtpPacket.cpp RtpPacket.h 

UniformQuantizer.o: UniformQuantizer.cpp UniformQuantizer.h readUInt.o readRealNumber.o

TypeList.o: TypeList.h TypeList.cpp CodeElement.h CodeElement.o

CodeElement.o: CodeElement.h CodeElement.cpp readUInt.h writeUInt.h DropList.h readUInt.o writeUInt.o

readRealNumber.o: readRealNumber.h readRealNumber.c

#testReadOne: testReadOne.c readUInt.o
readUInt.o: readUInt.h readUInt.c

readSInt.o: readSInt.h readSInt.c

#testWrite: testWrite.c writeUInt.o common_ljw.o

writeUInt.o: writeUInt.c writeUInt.h common_ljw.h

common_ljw.o: common_ljw.c common_ljw.h

.PHONY: doc
doc: cleandoc
	@$(MAKE) docfile

.PHONY: test 
test: 
	./readFile - -
	diff ../../../output/20140606_1719/1.case/foreman_cif_300.csvid ../../../output/20140606_1719/1.case/foreman_cif_ljw_300.csvid

.PHONY: docfile
docfile: 
	doxygen ./Doxyfile

.PHONY: cleandoc
cleandoc:
	rm -rf doc/html doc/*~

clean:
	rm *.o

