#include <json/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "json.h"
#include "cs_block.h"
#include "cs_config.h"
#include "cs_analysis.h"

#define CUDA_DBG 

void cs_config_getvalue( struct cs_config *csp, char *key, void *val ) ;

/*printing the value corresponding to boolean, double, integer and strings*/
void print_json_value
(json_object *jobj)
{
	enum json_type type;
	type = json_object_get_type(jobj); /*Getting the type of the json object*/
	// printf("type: ",type);
	switch (type) {
		case json_type_boolean: printf("json_type_boolean\n");
			printf("value: %s\n", json_object_get_boolean(jobj)? "true": "false");
			break;
		case json_type_double: printf("json_type_double\n");
			printf("          value: %lf\n", json_object_get_double(jobj));
			break;
		case json_type_int: printf("json_type_int\n");
			printf("          value: %d\n", json_object_get_int(jobj));
			break;
		case json_type_string: printf("json_type_string\n");
			printf("          value: %s\n", json_object_get_string(jobj));
			break;
	default:
	  assert(0);		// json_type_null or any other value should not happen
	}

}

void 
json_parse_array( struct cs_config *csp, json_object *jobj, char *key) 
{
	int json_parse( struct cs_config *csp, json_object * jobj); /*Forward Declaration*/
	enum json_type type;

	json_object *jarray = jobj; /*Simply get the array*/
	if(key) 
	{
		jarray = json_object_object_get(jobj, key); /*Getting the array if it is a key value pair*/
	}

#ifdef CUDA_OBS 
	printf("%s : key %s \n", __func__, key ) ;
#endif 

	int arraylen = json_object_array_length(jarray); /*Getting the length of the array*/
	// printf("Array Length: %d\n",arraylen);
	int i;
	json_object * jvalue;

	if (!( strcmp ( "interval_factor", key )))
	{
		if ( arraylen != CUBE_INFO_CNT )
		{
			printf("%s : missing element in \"interval\"\n", __func__ ) ;
			return ;
		}
		for (i=0; i< arraylen; i++)
		{
			jvalue = json_object_array_get_idx(jarray, i); /*Getting the array element at position i*/

			csp->interval_factor[i] = json_object_get_int( jvalue ) ;
		}

	} else
	{
		for (i=0; i< arraylen; i++)
		{
			jvalue = json_object_array_get_idx(jarray, i); /*Getting the array element at position i*/
			type = json_object_get_type(jvalue);
			if (type == json_type_array) {
				json_parse_array( csp, jvalue, NULL);
			}
			else if (type != json_type_object) {
				printf("value[%d]: ",i);
				print_json_value(jvalue);
			}
			else {
				json_parse( csp, jvalue);
			}
		}
	}
}

/*Parsing the json object*/
int
json_parse( struct cs_config *csp, json_object * jobj) 
{
	int i ;
	enum json_type type;
	json_object_object_foreach(jobj, key, val) 
	{ /*Passing through every array element*/
		type = json_object_get_type(val);
		// printf("key \"%s\" -- type: %d ",key, type);
		switch (type) {
		case json_type_boolean: 
		case json_type_double: 
			print_json_value(val);
			break ;

		case json_type_int: 
			i = json_object_get_int( val ) ;
			cs_config_getvalue ( csp, key, ( void *)&i) ;
			// print_json_value(val);
			break; 

		case json_type_string:
			cs_config_getvalue ( csp, key, ( void *)json_object_get_string( val )) ;
			// print_json_value(val);
			break; 
		case json_type_object:
			printf("json_type_object\n");
			jobj = json_object_object_get(jobj, key);
			json_parse( csp, jobj); 
			break;
		case json_type_array: 
			// printf("type: json_type_array, ");
			json_parse_array( csp, jobj, key);
			break;
		default:
		  assert(0); // json_type_null or any other value should not happen
		}
	}
	return ( 1 ) ;
} 

void 
cs_config_p( struct cs_config *csp )
{
	printf("cs_config_p: ------------------------------------------------------%p\n", csp ) ;
	
	printf("adj x %d y %d dbg %x do_perm %d perd %s\n",
		csp->adj_x,
		csp->adj_y,
		csp->dbg_flag,
		csp->do_permutation,
		csp->permdir );

	printf("video_src %s \n", csp->video_src ) ;

	// printf("finname %s\n", csp->finname ) ;

	printf("comp_ratio %d md %d %d %d cube %d %d %d do_cube %d\n",
		csp->comp_ratio,
		csp->md_x,
		csp->md_y,
		csp->md_z,
		csp->cubex,
		csp->cubey,
		csp->cubez, 
		csp->do_cube ) ;

	printf("frame exp %d %d %d edge %d %d display %d %d \n",
		csp->xadd,
		csp->yadd,
		csp->zadd,
		csp->edge_x,
		csp->edge_y,
		csp->disp_th_x,
		csp->disp_th_y ) ;

	// -d : frame size
	printf("frame size %d %d no seek %d md out %s\n",
		csp->frame_x,
		csp->frame_y,
		csp->do_not_seek,
		csp->md_outputfile ) ;

	printf("output file %s\n", csp->foutname ) ;

	printf("do one %d swap %d yonly %d overlap %d %d %d block %d %d %d\n",
		csp->do_one,
		csp->do_swap,
		csp->y_only,
		csp->overlap_x,
		csp->overlap_y,
		csp->overlap_z,
		csp->x_block,
		csp->y_block,
		csp->z_block ) ;

	printf("weight %d ana %d block %d comp %d interpo %d display_threshold %f\n",
		csp->weight_scheme,
		csp->do_analysis,
		csp->do_block,
		csp->do_comp_ratio,
		csp->do_interpolate,
	   	csp->display_threshold ) ;

	printf("reconstruction %d fps %d do_display %d ignore_edge %d capture %d\n",
		csp->do_reconstruction,
		csp->fps,
		csp->do_display,
		csp->ignore_edge,
	   	csp->capture ) ;

	printf("do_analysis %d interval %d %d %d \n",
		csp->do_analysis,
		csp->interval_factor[0],
		csp->interval_factor[1],
		csp->interval_factor[2] ) ;

	printf("cs_config_p: ------------------------------------------------------\n" ) ;
}

void 
cs_config_init( struct cs_config *csp )
{
	
	csp->adj_x = 0 ;
	csp->adj_y = 0 ;

	// -f : debug flag ;
	csp->dbg_flag = 0 ;

	// -p : do permutation 
	csp->do_permutation = 0 ;
	csp->permdir[0] = 0 ;

	// -I : ip cam string
	csp->video_src[0] = 0 ;
	csp->video_source = 0 ;

	// -i : input data file
	// csp->finname[0] = 0 ;

	csp->fps = 0 ;
	csp->do_display = 0 ;

	// -z : comp_ratio
	csp->comp_ratio = 0 ;

	// ignore edge: 1: yes ... i.e. no arrow for motion detection
	csp->ignore_edge = 0 ;

	// -m : motion detection
	csp->md_x = -1 ;
	csp->md_y = -1 ;
	csp->md_z = -1 ;

	// -c : cube
	csp->cubex = 0 ;
	csp->cubey = 0 ;
	csp->cubez = 0 ;

	csp->do_cube = 0 ; // derived

	// -e : frame expansion
	csp->xadd = 0 ;
	csp->yadd = 0 ;
	csp->zadd = 0 ;

	// -g : edge
	csp->edge_x = -1 ;
	csp->edge_y = -1 ;

	// -T : display threshold 
	csp->disp_th_x = 0 ;
	csp->disp_th_y = 0 ;

	// -d : frame size
	csp->frame_x = -1 ;
	csp->frame_y = -10 ;

	// -n : do not seek
	csp->do_not_seek = 0 ;

	// -F : md output file
	csp->md_outputfile[0] = 0 ;

	// -o : output file name
	csp->foutname[0] = 0 ;

	// -q
	csp->do_one = 0x7fffffff ;

	// -s
	csp->do_swap = 0 ;

	// -y
	csp->y_only = 0 ;

	// -O : overlap
	csp->overlap_x = 0 ;
	csp->overlap_y = 0 ;
	csp->overlap_z = 0 ;

	// -w : weight scheme
	csp->weight_scheme = NO_WEIGHT ;

	// -b : block size
	csp->x_block = -1 ;
	csp->y_block = -1 ;
	csp->z_block = -1 ;

	// reconstruction ;
	csp->do_reconstruction = 0.0 ;

	// capture
	csp->capture = 0 ;

	// display threshold
	csp->display_threshold = 0 ;

	// quantization

	csp->interval_factor[0] = 200 ;
	csp->interval_factor[1] = 200 ;
	csp->interval_factor[2] = 200 ;

	// global
	
	csp->do_analysis = 0 ;
	csp->do_block = 0	;
	csp->do_comp_ratio = 0 ;
	
	csp->do_interpolate = 1 ;
}

int
cs_config( char *jsonfile, struct cs_config *csp ) 
{
	int ret ;

	printf("cs_config: JSON string: %s\n", jsonfile ) ;
	json_object *jobj = json_object_from_file( jsonfile ) ;     

	ret = json_parse( csp, jobj);

	return ( ret ) ;
}

void
cs_config_getvalue( struct cs_config *csp, char *key, void *val )
{
	int i = *( int *)val ;
	char *charp = ( char *)val ;

	if ( !strcmp ( key, "adj_x" ))
	{
		csp->adj_x = i ;
	} else if ( !strcmp ( key, "adj_y" ))
	{
		csp->adj_y = i ;
	} else if ( !strcmp ( key, "dbg_flag" ))
	{
		if (( charp[0] == '0' ) && ( charp[1] == 'x' ))
			i = strtol ( charp, NULL, 16 ) ;
		else
			i = 0 ;
		csp->dbg_flag = i ;
	} else if ( !strcmp ( key, "do_display" ))
	{
		csp->do_display = i ;
	} else if ( !strcmp ( key, "fps" ))
	{
		csp->fps = i ;
	} else if ( !strcmp ( key, "do_permutation" ))
	{
		csp->do_permutation = i ;
	} else if ( !strcmp ( key, "comp_ratio" ))
	{
		csp->comp_ratio = i ;
	} else if ( !strcmp ( key, "md_x" ))
	{
		csp->md_x = i ;
	} else if ( !strcmp ( key, "md_y" ))
	{
		csp->md_y = i ;
	} else if ( !strcmp ( key, "md_z" ))
	{
		csp->md_z = i ;
	} else if ( !strcmp ( key, "doanalysis" ))
	{
		csp->do_analysis = i ;
	} else if ( !strcmp ( key, "cubex" ))
	{
		csp->do_cube++ ;
		csp->cubex = i ;
	} else if ( !strcmp ( key, "cubey" ))
	{
		csp->do_cube++ ;
		csp->cubey = i ;
	} else if ( !strcmp ( key, "cubez" ))
	{
		csp->do_cube++ ;
		csp->cubez = i ;
	} else if ( !strcmp ( key, "xadd" ))
	{
		csp->xadd = i ;
	} else if ( !strcmp ( key, "yadd" ))
	{
		csp->yadd = i ;
	} else if ( !strcmp ( key, "zadd" ))
	{
		csp->zadd = i ;
	} else if ( !strcmp ( key, "edge_x" ))
	{
		csp->edge_x = i ;
	} else if ( !strcmp ( key, "edge_y" ))
	{
		csp->edge_y = i ;
	} else if ( !strcmp ( key, "ignore_edge" ))
	{
		csp->ignore_edge = i ;
	} else if ( !strcmp ( key, "disp_th_x" ))
	{
		csp->disp_th_x = i ;
	} else if ( !strcmp ( key, "disp_th_y" ))
	{
		csp->disp_th_y = i ;
	} else if ( !strcmp ( key, "frame_x" ))
	{
		csp->frame_x = i ;
	} else if ( !strcmp ( key, "frame_y" ))
	{
		csp->frame_y = i ;
	} else if ( !strcmp ( key, "do_not_seek" ))
	{
		csp->do_not_seek = i ;
	} else if ( !strcmp ( key, "display_threshold" ))
	{
		csp->display_threshold = ( float )i / 100 ;
	} else if ( !strcmp ( key, "do_one" ))
	{
		csp->do_one = i ;
	} else if ( !strcmp ( key, "do_swap" ))
	{
		csp->do_swap = i ;
	} else if ( !strcmp ( key, "y_only" ))
	{
		csp->y_only = i ;
	} else if ( !strcmp ( key, "overlap_x" ))
	{
		csp->do_block++ ;
		csp->overlap_x = i ;
	} else if ( !strcmp ( key, "overlap_y" ))
	{
		csp->do_block++ ;
		csp->overlap_y = i ;
	} else if ( !strcmp ( key, "overlap_z" ))
	{
		csp->do_block++ ;
		csp->overlap_z = i ;
	} else if ( !strcmp ( key, "weight_scheme" ))
	{
		csp->do_block++ ;
		csp->weight_scheme = i ;
	} else if ( !strcmp ( key, "x_block" ))
	{
		csp->do_block++ ;
		csp->x_block = i ;
	} else if ( !strcmp ( key, "y_block" ))
	{
		csp->do_block++ ;
		csp->y_block = i ;
	} else if ( !strcmp ( key, "z_block" ))
	{
		csp->do_block++ ;
		csp->z_block = i ;
	} else if ( !strcmp ( key, "capture"))
	{
		csp->capture = i ;
	} else if ( !strcmp ( key, "reconstruction"))
	{
		csp->do_reconstruction = i ;
	} else if ( !strcmp ( key, "video_source"))
	{
		csp->video_source = i ;
	// string ...

	} else if ( !strcmp ( key, "video_src" ))
	{
		i = strlen ( charp ) ;
		if ( i < PATH_LENG )
			strcpy ( csp->video_src, charp ) ;
		else
			printf("video_src too long %d\n", i ) ;

	} else if ( !strcmp ( key, "permdir" ))
	{
		i = strlen ( charp ) ;
		if ( i < PATH_LENG )
			strcpy ( csp->permdir, charp ) ;
		else
			printf("permdir too long %d\n", i ) ;

#ifdef CUDA_OBS 
	} else if ( !strcmp ( key, "finname" ))
	{
		i = strlen ( charp ) ;
		if ( i < PATH_LENG )
			strcpy ( csp->finname, charp ) ;
		else
			printf("finname too long %d\n", i ) ;
#endif 

	} else if ( !strcmp ( key, "md_outputfile" ))
	{
		i = strlen ( charp ) ;
		if ( i < PATH_LENG )
			strcpy ( csp->md_outputfile, charp ) ;
		else
			printf("permdir too long %d\n", i ) ;

	} else if ( !strcmp ( key, "foutname" ))
	{
		i = strlen ( charp ) ;
		if ( i < PATH_LENG )
			strcpy ( csp->foutname, charp ) ;
		else
			printf("foutname too long %d\n", i ) ;

	} else
	{
		printf("wrong key %s \n", key ) ;
	}
}
