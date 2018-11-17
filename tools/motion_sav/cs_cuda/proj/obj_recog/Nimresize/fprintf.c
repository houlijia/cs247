/*
 * fprintf.c
 *
 * Code generation for function 'fprintf'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "fprintf.h"
#include "fileManager.h"
#include <stdio.h>

/* Function Declarations */
static double c_fprintf(void);
static double e_fprintf(void);

/* Function Definitions */
static double c_fprintf(void)
{
  int nbytesint;
  FILE * b_NULL;
  boolean_T autoflush;
  FILE * filestar;
  static const char cfmt[52] = { 'I', 'n', 'p', 'u', 't', ' ', 'i', 's', ' ',
    't', 'o', 'o', ' ', 's', 'm', 'a', 'l', 'l', ' ', 'f', 'o', 'r', ' ', 'b',
    'i', 'l', 'i', 'n', 'e', 'a', 'r', ' ', 'o', 'r', ' ', 'b', 'i', 'c', 'u',
    'b', 'i', 'c', ' ', 'm', 'e', 't', 'h', 'o', 'd', ';', '\x0a', '\x00' };

  nbytesint = 0;
  b_NULL = NULL;
  fileManager(&filestar, &autoflush);
  if (filestar == b_NULL) {
  } else {
    nbytesint = fprintf(filestar, cfmt);
    fflush(filestar);
  }

  return nbytesint;
}

static double e_fprintf(void)
{
  int nbytesint;
  FILE * b_NULL;
  boolean_T autoflush;
  FILE * filestar;
  static const char cfmt[40] = { 'u', 's', 'i', 'n', 'g', ' ', 'n', 'e', 'a',
    'r', 'e', 's', 't', '-', 'n', 'e', 'i', 'g', 'h', 'b', 'o', 'r', ' ', 'm',
    'e', 't', 'h', 'o', 'd', ' ', 'i', 'n', 's', 't', 'e', 'a', 'd', '.', '\x0a',
    '\x00' };

  nbytesint = 0;
  b_NULL = NULL;
  fileManager(&filestar, &autoflush);
  if (filestar == b_NULL) {
  } else {
    nbytesint = fprintf(filestar, cfmt);
    fflush(filestar);
  }

  return nbytesint;
}

void b_fprintf(void)
{
  c_fprintf();
}

void d_fprintf(void)
{
  e_fprintf();
}

/* End of code generation (fprintf.c) */
