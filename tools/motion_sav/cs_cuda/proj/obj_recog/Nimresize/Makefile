.PHONY: all clean
all: all.o any.o conv2.o fileManager.o filter2.o floor.o fprintf.o meshgrid.o \
	Nimresize_emxAPI.o Nimresize_emxutil.o Nimresize_initialize.o Nimresize.o \
	Nimresize_terminate.o rdivide.o rot90.o rtGetInf.o rtGetNaN.o rt_nonfinite.o


H_DIRS=.

%.o : %.c
	g++ -I. -I$(H_DIRS) -c $<

clean:
	rm -f *.o
