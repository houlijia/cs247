#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>

#include "cs_header.h"

void
pusage( char *s )
{
	fprintf( stderr, "Usage: %s measurement-file-name\n", s ) ;
}

static struct cs_header csh ;

const char *coding[] = {
	"YUV420P"
} ;

static const char SIZE_OF_CODING = char(sizeof( coding ) / sizeof ( void *));

const char *yuv420p_coding_opt[] = {
	"Y-ONLY",
	"DOUBLE_PERM",
	"ML_SEQ",
	"WEIGHT",
	"OVERLAP",
	"N/A",
	"N/A",
	"BIG-ENDIAN"
} ;

const char *matrix[] = {
	"Walsh Hardaman"
} ;

static const char SIZE_OF_MATRIX = char(sizeof( matrix ) / sizeof ( void *));

void
poption( const char *s, const char *o[], char opt )
{
	int p1 = 0, first = 1, i = 0 ;

	printf("%s", s );
	while ( opt )
	{
		if ( opt & ( 1 << i ))
		{
			p1++ ;
			if ( first )
			{
				printf("%s", o[i] ) ;
				first = 0 ;
			} else
				printf(" | %s", o[i] ) ;

			opt &= ~( 1 << i ) ;
		}
		i++ ;
	}

	printf("\n") ;
}

main( int ac, char *av[] )
{
	int i, fin ;

	if ( ac < 2 )
	{
		pusage( av[0] ) ;
		exit( 1 ) ;
	}

	fin = open( av[1], O_RDONLY ) ;

	if ( fin < 0 )
	{
		fprintf( stderr, "open file %s failed errno %d\n", av[1], errno ) ;
		pusage( av[0] ) ;
		exit( 1 ) ;
	}

	if (!cs_get_header ( fin, &csh ))
	{
		fprintf( stderr, "read file %s failed \n", av[1] ) ;
		pusage( av[0] ) ;
		exit( 1 ) ;
	}

	if (( csh.coding > 0 ) && ( csh.coding <= SIZE_OF_CODING ))
		printf("orig coding:	%s\n", coding[ csh.coding - 1 ] ) ;
	else
		printf("orig coding:	%d\n", csh.coding ) ;

	if ( csh.coding == CS_CD_YUV420P )
		poption("coding_opt:	", yuv420p_coding_opt, csh.coding_opt ) ;

	if (( csh.matrix > 0 ) && ( csh.matrix <= SIZE_OF_MATRIX ))
		printf("matrix:		%s\n", matrix[ csh.matrix - 1 ] ) ;
	else
		printf("matrix:		%d\n", csh.matrix ) ;

	printf("frame x/y:	%d %d\nblock x/y/z: 	%d %d %d\n",
		csh.frame.x, csh.frame.y, 
		csh.block.x, csh.block.y, csh.block.z ) ;

	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
		printf("cube %d x/y/z: %d %d %d\n",
			i, csh.select[i].x, csh.select[i].y, csh.select[i].z ) ;

	printf("overlap x/y/z: %d %d %d\n",
		csh.overlap.x, csh.overlap.y, csh.overlap.z ) ;

	printf("expansion x/y/z: %d %d %d -- append x/y: %d %d\n",
		csh.expand.x, csh.expand.y, csh.expand.z,
		csh.append.x, csh.append.y ) ;

	printf("motion x/y/z: %d %d %d -- edge x/y: %d %d\n",
		csh.md.x, csh.md.y, csh.md.z,
		csh.edge.x, csh.edge.y ) ;

	printf("weight scheme %d\n",
		csh.weight ) ;
}
