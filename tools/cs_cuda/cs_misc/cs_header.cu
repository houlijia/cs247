#include <stdio.h>
#include <errno.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "cs_header.h"

int 
cs_put_header( int fd, char coding, char opt, char m,
	int x, int y,
	int xb, int yb, int zb,
	int xc0, int yc0, int zc0,
	int xc1, int yc1, int zc1,
	int xc2, int yc2, int zc2,
	int xo, int yo, int zo,
	int xe, int ye, int ze,
	int xa, int ya,
	int edge_x, int edge_y,
	int md_x, int md_y, int md_z,
	int weight )
{
	int i ;
	struct cs_header csh ;

	memset( &csh, 0, sizeof( csh )) ;

	csh.coding = coding ;
	csh.coding_opt = opt ;
	csh.matrix = m ;

	csh.frame.x = htonl( x );
	csh.frame.y = htonl( y );

	csh.block.x = htonl( xb );
	csh.block.y = htonl( yb );
	csh.block.z = htonl( zb );

	csh.select[0].x = htonl( xc0 );
	csh.select[0].y = htonl( yc0 );
	csh.select[0].z = htonl( zc0 );

	csh.select[1].x = htonl( xc1 );
	csh.select[1].y = htonl( yc1 );
	csh.select[1].z = htonl( zc1 );

	csh.select[2].x = htonl( xc2 );
	csh.select[2].y = htonl( yc2 );
	csh.select[2].z = htonl( zc2 );

	csh.overlap.x = htonl( xo );
	csh.overlap.y = htonl( yo );
	csh.overlap.z = htonl( zo );

	csh.expand.x = htonl( xe );
	csh.expand.y = htonl( ye );
	csh.expand.z = htonl( ze );

	csh.md.x = htonl( md_x );
	csh.md.y = htonl( md_y );
	csh.md.z = htonl( md_z );

	csh.edge.x = htonl( edge_x );
	csh.edge.y = htonl( edge_y );

	csh.append.x = htonl( xa );
	csh.append.y = htonl( ya );

	csh.weight = htons( weight ) ;

	i = write ( fd, &csh, sizeof ( csh )) ;

	if ( i != sizeof ( csh ))
	{
		fprintf( stderr, "cs_put_header failed errno %d\n", errno ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}

int 
cs_get_header( int fd, struct cs_header *cshp )
{
	int i ;

	i = read ( fd, cshp, sizeof ( *cshp )) ;

	if ( i != sizeof ( *cshp ))
	{
		fprintf( stderr, "cs_get_header failed errno %d\n", errno ) ;
		return ( 0 ) ;
	}

	cshp->frame.x = ntohl( cshp->frame.x );
	cshp->frame.y = ntohl( cshp->frame.y );

	cshp->block.x = ntohl( cshp->block.x );
	cshp->block.y = ntohl( cshp->block.y );
	cshp->block.z = ntohl( cshp->block.z );

	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		cshp->select[i].x = ntohl( cshp->select[i].x );
		cshp->select[i].y = ntohl( cshp->select[i].y );
		cshp->select[i].z = ntohl( cshp->select[i].z );
	}

	cshp->overlap.x = ntohl( cshp->overlap.x );
	cshp->overlap.y = ntohl( cshp->overlap.y );
	cshp->overlap.z = ntohl( cshp->overlap.z );

	cshp->expand.x = ntohl( cshp->expand.x );
	cshp->expand.y = ntohl( cshp->expand.y );
	cshp->expand.z = ntohl( cshp->expand.z );

	cshp->append.x = ntohl( cshp->append.x );
	cshp->append.y = ntohl( cshp->append.y );

	cshp->md.x = ntohl( cshp->md.x );
	cshp->md.y = ntohl( cshp->md.y );
	cshp->md.z = ntohl( cshp->md.z );

	cshp->edge.x = ntohl( cshp->edge.x );
	cshp->edge.y = ntohl( cshp->edge.y );

	cshp->weight = ntohs( cshp->weight ) ;

	return ( 1 ) ;
}

int 
cs_put_block_header( int fd, int random_l, int random_r )
{
	int i ;
	struct cs_block_header csh ;

	memset( &csh, 0, sizeof( csh )) ;

	csh.random_l = htonl( random_l ) ;
	csh.random_r = htonl( random_r ) ;

	i = write ( fd, &csh, sizeof ( csh )) ;

	if ( i != sizeof ( csh ))
	{
		fprintf( stderr, "cs_put_block_header failed errno %d\n", errno ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}

int 
cs_get_block_header( int fd, struct cs_block_header *cshp )
{
	int i ;

	i = read ( fd, cshp, sizeof ( *cshp )) ;

	if ( i != sizeof ( *cshp ))
	{
		fprintf( stderr, "cs_get_block_header failed errno %d\n", errno ) ;
		return ( 0 ) ;
	}

	cshp->random_l = ntohl( cshp->random_l );
	cshp->random_r = ntohl( cshp->random_r );

	return ( 1 ) ;
}
