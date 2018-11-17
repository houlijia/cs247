/* Copyright (C) 2012 Alcatel-Lucent */

/** \file gnrt_lfsr.c */

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include <arpa/inet.h>

#include "parse_opt.h"
#include "prim_poly.h"

static int WH_write(bitpol_t L_val, bitpol_t R_val);

static unsigned long verbose = 0;
static unsigned long max_degree = 0;
static const char *prim_poly_fname = "primpoly.dat";
static const char *lfsr_seq_fname="lfsr_%d_seq.dat";
static const char *WH_indx_fname="WH_lfsr_%d_indcs.dat";
static FILE *WH_output = NULL;
static WH_write_fun WH_write_ptr = NULL;

static void help(const char *command)
{
    printf("Compute maximum lenght sequences\n"
	   "  %s [OPTIONS] max_degree \n\n"
	   "Options:\n"
	   "-h - print help message\n"
	   "-i<prim_poly_file> - path to a file containing primitive polynomials\n"
	   "                     Default: \"%s\"\n"
	   "-o<ML_seq_file_template> - template for file names for ML sequences.\n"
	   "        The template should contain the string \"%%d\", which is replaced by\n"
	   "        the degree of the polynomial. Default \"%s\"\n"
	   "-v<verbosity> - verbosity level. Default=%lu\n"
	   "-w<WH_indices_template> - If present, another set of files is generated\n"
	   "        which contains indexing into the Walsh-Hadamrad matrix of order\n"
	   "        2^D (D=polynomial degree).  The template should contain the string\n"
	   "        \"%%d\", which is replaced by D. if the specified name is \".\",\n"
	   "        the template1 \"%s\" is used.\n"
	   "        Each of the generated files contains pairs L(i),R(i),\n"
	   "        0<=i<(2^D-1). Each pair member is a %lu-byte unsigned integer in\n"
	   "        network order. Let H(i,j) be the entries of the (Hadamard order)\n"
	   "        Walsh-Hadamard matrix,  0<=i,j<2^D and let s(0),s(1),... be\n"
	   "        the computed ML sequence.  Then\n"
	   "          H(L(i),R(j)) = s(i-1-j). \n"
	   "\n\n"
	   "max_degree - The maximal degree for which sequences are computed.\n"
	   "        However, a sequence will not be computed if a polynomial of the\n"
	   "        needed degree is not available in prim_poly_file\n"
	   ,
	   command, prim_poly_fname, lfsr_seq_fname, verbose,WH_indx_fname,
	   (unsigned long)sizeof(bitpol_t));
}

int main(int argc, char **argv)
{
    ParseOpt papt = PARSE_OPT_INIT("hi:o:v:w:",1);
    PrimPolyList prlist = PrimPolyListInitializer;
    char err_msg[256];
    int c, k;
    char *nxt;
    unsigned long val;
    int err;


    /* Parse options */
    while((c=nextParseOpt(argc,argv,&papt)) != -1) {

	switch(c) {
	case 'h':
	    help(argv[0]);
	    exit(EXIT_SUCCESS);
	case 'i':
	    prim_poly_fname = papt.arg;
	    break;
	case 'o':
	    if(!strcmp(papt.arg, "-"))
		lfsr_seq_fname = NULL;
	    else
		lfsr_seq_fname = papt.arg;
	    break;
	case 'v':
	    val = strtoul(papt.arg,&nxt,0);
	    if(*nxt != '\0') {
		fprintf(stderr, "Illegal value for option -v: \"%s\"", papt.arg);
		exit(EXIT_FAILURE);
	    }
	    verbose = val;
	    break;

	case 'w':
	    WH_write_ptr = WH_write;
	    if(strcmp(papt.arg, "."))
		WH_indx_fname = papt.arg;
	    break;
		
	default:
	    fprintf(stderr, "Unexpected option: -%c\n", c);
	    exit(EXIT_FAILURE);
	}
    }

    switch(argc - papt.ind) {
    case 0:
	fprintf(stderr, "Too few arguments\n");
	exit(EXIT_FAILURE);
    case 1:
	val = strtoul(argv[papt.ind],&nxt,0);
	if(*nxt != '\0') {
	    fprintf(stderr, "Illegal value for degree: \"%s\"", papt.arg);
	    exit(EXIT_FAILURE);
	}
	max_degree = val;
	break;
    default:
	fprintf(stderr, "Too many arguments\n");
	exit(EXIT_FAILURE);
    }

    c = readPrimPolyList(prim_poly_fname, &prlist, err_msg, sizeof(err_msg)); 

    if(c) {
	fprintf(stderr, "Reading %s failed: %s\n", prim_poly_fname, err_msg);
	exit(EXIT_FAILURE);
    }

    for(k=0; k<prlist.cnt; k++) {
	/* Loop on degrees */
	PrimPoly *prmply = &prlist.prpl[k];
	size_t j;
	char wh_fname[1024];

	if(max_degree && prmply->degree > max_degree)
	    continue;

	if(verbose)
	    printf("\n%d 0x%0X", prmply->degree, prmply->coefs);

	if (WH_write_ptr != NULL) {
	    sprintf(wh_fname, WH_indx_fname, prmply->degree);
	    WH_output = fopen(wh_fname, "w");
	    if(WH_output == NULL) {
		fprintf(stderr, "degree %d, failed opening WH output file %s (error %d: %s)\n",
			prmply->degree, wh_fname, errno, strerror(errno));
		free(prmply->seq);
		prmply->seq = NULL;
		break;
	    }
	}

	prmply->seq = comp_lfsrPrimPoly(prmply, 1, err_msg, sizeof(err_msg));
	if(prmply->seq == NULL) {
	    fprintf(stderr, "Error in computing sequence: %s\n", err_msg);
	    break;
	}

	if(verbose >= 2) {
	    u_8 mask=1;
	    size_t indx = 0;
      
	    printf(":");

	    for (j=0; j<orderPrimPoly(prmply); j++) {
		printf("%s%u", (j%80==0)?"\n\t":"", !!(prmply->seq[indx]&mask));
		mask <<=1;
		if(!mask) {
		    mask = 1;
		    indx++;
		}
	    }
	    printf("\n");
	}

	if(lfsr_seq_fname != NULL) {
	    char fname[1024];
	    size_t n_byte = ((1<<prmply->degree)-1)/8+1;
	    FILE *out;

	    sprintf(fname, lfsr_seq_fname, prmply->degree);

	    out = fopen(fname, "wb");
	    if(out == NULL) {
		fprintf(stderr, "degree %d, failed opening output file %s (error %d: %s)\n",
			prmply->degree, wh_fname, err, strerror(err));
		free(prmply->seq);
		prmply->seq = NULL;
		break;
	    }

	    c = fwrite(prmply->seq, sizeof(u_8), n_byte, out);
	    if(c != n_byte) {
		err = ferror(out);
		fprintf(stderr,"Failed writing %s (error %d: %s)\n", fname, err,
			strerror(err));
		free(prmply->seq);
		prmply->seq = NULL;
		return err;
	    }

	    fclose(out);


	}

	if (WH_write_ptr != NULL) {
	    err = comp_lfsr_wh(prmply, WH_write_ptr);
	    if(err) {
		fprintf(stderr, "Faliled writing %s (error %d: %s)\n",
			wh_fname, err, strerror(err));
		free(prmply->seq);
		prmply->seq = NULL;
		return err;
	    }
	    fclose(WH_output);
	}

	free(prmply->seq);
	prmply->seq = NULL;
    }

    free(prlist.prpl);

    return EXIT_SUCCESS;
}

static int WH_write(bitpol_t L_val, bitpol_t R_val)
{
    const bitpol_t vals[2] = { htonl(L_val), htonl(R_val)};
    int cnt;
    int err = 0;
    
    cnt = fwrite(vals, sizeof(vals), 1, WH_output);
    if(cnt != 1) {
	err = errno;
	errno = 0;
    }

    return err;
}
