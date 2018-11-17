#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

// change the suspected u & v field to 0 ...

#define BUF_SIZE	( 3 * 10240 ) 
#define YUV_SIZE	6 // 4 pixels

void fix_it ( int fin, int fout ) ;
char *buf ;

main( int ac, char *av[] )
{
	int fin, fout ;

	if ( ac < 3 )
	{
		printf("Usage: %s yuv-file-in yuv-file-out\n", av[0]) ;
		exit( 1 ) ;
	}

	buf = ( char * ) malloc ( BUF_SIZE ) ;

	if ( buf == NULL )
	{
		printf("malloc failed \n") ;
		exit ( 1 ) ;
	}

	fin = open( av[1], O_RDONLY ) ;

	if ( fin == -1 )
	{
		printf("file %s does not exist\n", av[1]) ;
		printf("Usage: %s yuv-file-in yuv-file-out\n", av[0]) ;
		exit( 1 ) ;
	}

	fout = open( av[2], O_CREAT | O_TRUNC | O_RDWR ) ;

	if ( fout == -1 )
	{
		printf("file %s open failed %d\n", av[1], errno ) ;
		printf("Usage: %s yuv-file-in yuv-file-out\n", av[0]) ;
		exit( 1 ) ;
	}

	fix_it( fin, fout ) ;

	close ( fin ) ;
	close ( fout ) ;

}

void
fix_it ( int fin, int fout )
{
	int i, yuv_cnt, total, osize, size ;
	char *cp ;

	total = 0 ;
	yuv_cnt = 1 ;
	while ( 1 )
	{
		if (( size = read ( fin, buf, BUF_SIZE )) <= 0 )
		{
			printf("fix_it: overall size %d\n", total ) ;
			return ;
		}

		// printf("read %d total %d \n", size, total ) ;

		if ( size % YUV_SIZE )
			printf("fix_it: wrong size %d total %d \n", size, total ) ;

		osize = size ;
		cp = buf ;
		total += size ;
		while ( size-- )
		{
			if (!( yuv_cnt % 5 ))
				*cp = 0 ;
			else if (!( yuv_cnt % 6 ))
				*cp = 0 ;

#ifdef HCD_OBS 
			if (!( yuv_cnt % 3 ))
				*cp = 0 ;
#endif 
			yuv_cnt++ ;

			cp++ ;
		}

		if (( i = write ( fout, buf, osize )) != osize )
			printf("fix_it: failed %d %d %d\n", errno, i, fout )	;
	}

	printf("fix_it: overall size %d\n", total ) ;
}
