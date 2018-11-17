all : prank dataskip datar2c dataa2b datab2a idx whm2 whm whmt datagen datagenb

dataskip : dataskip.c 
	cc dataskip.c -g -o dataskip

dataa2b : dataa2b.c 
	cc dataa2b.c -g -o dataa2b

datar2c : datar2c.c 
	cc datar2c.c -g -o datar2c

datab2a : datab2a.c 
	cc datab2a.c -g -o datab2a

datagenb : datagenb.c 
	cc datagenb.c -g -o datagenb

datagen : datagen.c 
	cc datagen.c -g -o datagen

whmt : whmt.c 
	cc whmt.c -g -lm -o whmt

idx : idx.c 
	cc idx.c -g -lm -o idx

whm : whm.c 
	cc whm.c -g -lm -o whm

whm2 : whm2.c 
	cc whm2.c -g -lm -o whm2

prank : prank.c 
	cc prank.c -g -lm -o prank

clean :
	rm whm
	rm whmt
	rm datagen
	rm datagenb
	rm datab2a
	rm dataa2b
	rm datar2c
	rm idx
	rm whm2
	rm prank
