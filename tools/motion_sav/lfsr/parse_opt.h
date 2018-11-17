/* Copyright Alcatel-Lucent 2006 */

/** \file parse_opt.h Declartions of channels specifications

*/

#ifndef PARSE_OPT_H_GRD
#define PARSE_OPT_H_GRD 

#ifndef EXT_C
#ifdef __cplusplus
#define EXT_C extern "C"
#else
#define EXT_C
#endif
#endif

/* Alternative to getopt */
/* This struct is used for thread safe option parsing (instead of getopt()). */
typedef struct ParseOpt {
  const char *opt;		/* a string of option. ':' following an option indicates it has an argument */
  int ind;			/* index of next argument */
  char *arg;                    /* pointer to argument of next option, or NULL if there are none */
  char *next_opt;		/* If not NULL, A pointer to the next letter argument (multiple letter arg)  */
} ParseOpt;

#define PARSE_OPT_INIT(opt_str,indx) {opt_str, indx, NULL, NULL}

EXT_C int
nextParseOpt(int argc,
             char *argv[],
             struct ParseOpt *po
             );                                 /* returns option letter if successful or -1 otherwise */

#endif  /* PARSE_OPT_H_GRD */

