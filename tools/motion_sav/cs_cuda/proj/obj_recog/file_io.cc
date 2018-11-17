#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

int
file_in( const char *filename, int size, char *bufp )
{
	int i, fid ;

	fid = open ( filename, O_RDONLY ) ;

	if ( fid < 0 )
	{
		printf("%s : open failed %d \n", __func__, errno ) ;
		return ( 0 ) ;
	}

	i = read( fid, bufp, size ) ;

	if ( i != size )
	{
		printf("%s : read failed got %d want %d errno %d \n", __func__, i, size, errno ) ;
		close ( fid ) ;
		return ( 0 ) ;
	}
	close( fid ) ;
	return ( 1 ) ;
}

int
file_out( const char *filename, int size, char *bufp )
{
	int i, fid ;

	fid = open ( filename, O_WRONLY | O_TRUNC | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO ) ;

	if ( fid < 0 )
	{
		printf("%s : open failed %d \n", __func__, errno ) ;
		return ( 0 ) ;
	}

	i = write( fid, bufp, size ) ;

	if ( i != size )
	{
		printf("%s : write failed got %d want %d errno %d \n", __func__, i, size, errno ) ;
		close ( fid ) ;
		return ( 0 ) ;
	}
	close( fid ) ;
	return ( 1 ) ;
}




