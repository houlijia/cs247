all: file_io.o resize_test 

file_io.o : file_io.c
	cc -c file_io.c

resize_test: main.c main.h file_io.o
	cc main.c -I. -I.. ../*.o file_io.o -lm -o resize_test

clean:
	rm file_io.o
	rm resize_test
