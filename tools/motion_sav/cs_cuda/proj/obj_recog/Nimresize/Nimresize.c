/*
 * Nimresize.c
 *
 * Code generation for function 'Nimresize'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "Nimresize_emxutil.h"
#include "meshgrid.h"
#include "floor.h"
#include "rdivide.h"
#include "all.h"
#include "filter2.h"
#include "fprintf.h"
#include "any.h"
#include <stdio.h>

/* Function Declarations */
static void resizeImage(const double A[307200], const double m[2],
  emxArray_real_T *b);
static double rt_roundd_snf(double u);

/* Function Definitions */
static void resizeImage(const double A[307200], const double m[2],
  emxArray_real_T *b)
{
  char method[3];
  int i2;
  static const char cv0[3] = { 'b', 'i', 'c' };

  double bsize[2];
  boolean_T b_bsize[2];
  int k;
  double kd;
  static const char cv1[3] = { 'n', 'e', 'a' };

  int nm1d2;
  double h1_data[11];
  static const double dv0[11] = { -2.4963101798827729E-18,
    -0.0078558540950236788, 0.0401711752634344, -0.10331480155755128,
    0.17076215646943421, 0.80047464783941269, 0.17076215646943421,
    -0.10331480155755128, 0.0401711752634344, -0.0078558540950236788,
    -2.4963101798827729E-18 };

  int h2_size[2];
  double h2_data[11];
  static double dv1[307200];
  double b_h1_data[11];
  int h1_size[1];
  static double a[307200];
  emxArray_real_T *uu;
  emxArray_real_T *vv;
  double dx;
  double dy;
  double t;
  int n;
  double anew;
  double apnd;
  double ndbl;
  double cdiff;
  boolean_T b_method[3];
  static const char cv2[3] = { 'b', 'i', 'l' };

  boolean_T guard1 = false;
  double b_vv[2];
  double dv2[2];
  unsigned int c_vv[2];
  double nrem[2];
  emxArray_real_T *rows;
  double mblocks;
  double nblocks;
  int i;
  emxArray_real_T *cols;
  emxArray_real_T *v;
  emxArray_int32_T *r0;
  emxArray_int32_T *r1;
  emxArray_real_T *r2;
  emxArray_real_T *minval;
  emxArray_real_T *b_uu;
  emxArray_real_T *d_vv;
  int j;
  static double VV[309444];
  int nout;
  for (i2 = 0; i2 < 3; i2++) {
    method[i2] = cv0[i2];
  }

  /* else */
  /* rout = zeros([size(r),3]); */
  /* rout(:,:,1) = r; */
  /* rout(:,:,2) = g; */
  /* rout(:,:,3) = b; */
  /* end */
  /* else % nargout==3 */
  /* if strcmp(classIn,'uint8') */
  /* rout = uint8(round(r*255));  */
  /* g = uint8(round(g*255));  */
  /* b = uint8(round(b*255));  */
  /* else */
  /* rout = r;        % g,b are already defined correctly above */
  /* end */
  /* end */
  /* else  */
  /* r = resizeImage(A,m,method,h); */
  /* if nargout==0, */
  /*       %% LDL imshow(r); */
  /* return; */
  /* end */
  /* if strcmp(classIn,'uint8') */
  /* r = uint8(round(r*255));  */
  /* end */
  /* rout = r; */
  /* end */
  /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
  /*  */
  /*   Function: resizeImage */
  /*  */
  /*  Inputs: */
  /*          A       Input Image */
  /*          m       resizing factor or 1-by-2 size vector */
  /*          method  'nearest','bilinear', or 'bicubic' */
  /*          h       the anti-aliasing filter to use. */
  /*                  if h is zero, don't filter */
  /*                  if h is an integer, design and use a filter of size h */
  /*                  if h is empty, use default filter */
  /* if prod(size(m))==1, */
  /* bsize = floor(m*size(A)); */
  /* else */
  /* end */
  /*  values in bsize must be at least 1. */
  for (k = 0; k < 2; k++) {
    if (m[k] >= 1.0) {
      kd = m[k];
    } else {
      kd = 1.0;
    }

    b_bsize[k] = ((kd < 4.0) && (kd < 480.0 + 160.0 * (double)k));
    bsize[k] = kd;
  }

  if (any(b_bsize)) {
    b_fprintf();
    d_fprintf();
    for (i2 = 0; i2 < 3; i2++) {
      method[i2] = cv1[i2];
    }
  }

  /*  Default filter size */
  if (method[0] == 'b') {
    /*  Design anti-aliasing filter if necessary */
    /* %%if bsize(1)<m, h1="fir1(nn-1,bsize(1)/m);" else="" end="" if="" bsize(2)<n,="" h2="fir1(nn-1,bsize(2)/n    );" %="" length(h1)="">1 | length(h2)&gt;1, h = h1'*h2; else h = []; end */
    /* if bsize(1)<m, */
    /* h=1fir1(nn-1,bsize(1)/m); */
    /* else  */
    /* end  */
    /* if bsize(2)<n,  */
    /* h2="fir1(nn-1,bsize(2)/n); */
    /* if length(h1)>1 | length(h2)>1, */
    /* h = h1'*h2; */
    /* else  */
    /* h = [];  */
    /* end */
    /*  ---------------------------------- */
    if (bsize[0] < 480.0) {
      nm1d2 = 11;
      memcpy(&h1_data[0], &dv0[0], 11U * sizeof(double));

      /* % LDL place holder ... will not work ... 512/744 will */
      /* h1=fir1(nn-1,bsize(1)/m); */
    } else {
      nm1d2 = 1;
      h1_data[0] = 1.0;

      /* h1=[] % LDL added */
    }

    if (bsize[1] < 640.0) {
      h2_size[0] = 1;
      h2_size[1] = 11;
      memcpy(&h2_data[0], &dv0[0], 11U * sizeof(double));

      /* % LDL  */
      /* h2=fir1(nn-1,bsize(2)/n);  */
    } else {
      h2_size[0] = 1;
      h2_size[1] = 1;
      h2_data[0] = 1.0;

      /* h2=[] % LDL added */
    }

    /*  ---------------------------------- */
    if ((nm1d2 > 1) || (h2_size[1] > 1)) {
      filter2(h2_data, h2_size, A, dv1);
      h1_size[0] = nm1d2;
      for (i2 = 0; i2 < nm1d2; i2++) {
        b_h1_data[i2] = h1_data[i2];
      }

      b_filter2(b_h1_data, h1_size, dv1, a);
    } else {
      memcpy(&a[0], &A[0], 307200U * sizeof(double));
    }
  } else {
    memcpy(&a[0], &A[0], 307200U * sizeof(double));
  }

  emxInit_real_T(&uu, 2);
  emxInit_real_T(&vv, 2);
  if (method[0] == 'n') {
    /*  Nearest neighbor interpolation */
    dx = 640.0 / bsize[1];
    dy = 480.0 / bsize[0];
    t = dx / 2.0;
    kd = dx / 2.0 + 0.5;
    if (rtIsNaN(kd) || rtIsNaN(dx)) {
      n = 0;
      anew = rtNaN;
      apnd = 640.5;
    } else if (dx == 0.0) {
      n = -1;
      anew = t + 0.5;
      apnd = 640.5;
    } else if (rtIsInf(kd)) {
      n = 0;
      anew = rtNaN;
      apnd = 640.5;
    } else if (rtIsInf(dx)) {
      n = 0;
      anew = t + 0.5;
      apnd = 640.5;
    } else {
      anew = t + 0.5;
      ndbl = floor((640.5 - (t + 0.5)) / dx + 0.5);
      apnd = (t + 0.5) + ndbl * dx;
      if (dx > 0.0) {
        cdiff = apnd - 640.5;
      } else {
        cdiff = 640.5 - apnd;
      }

      kd = fabs(t + 0.5);
      if (kd >= 640.5) {
      } else {
        kd = 640.5;
      }

      if (fabs(cdiff) < 4.4408920985006262E-16 * kd) {
        ndbl++;
        apnd = 640.5;
      } else if (cdiff > 0.0) {
        apnd = (t + 0.5) + (ndbl - 1.0) * dx;
      } else {
        ndbl++;
      }

      if (ndbl >= 0.0) {
        n = (int)ndbl - 1;
      } else {
        n = -1;
      }
    }

    i2 = uu->size[0] * uu->size[1];
    uu->size[0] = 1;
    uu->size[1] = n + 1;
    emxEnsureCapacity((emxArray__common *)uu, i2, (int)sizeof(double));
    if (n + 1 > 0) {
      uu->data[0] = anew;
      if (n + 1 > 1) {
        uu->data[n] = apnd;
        nm1d2 = (n + (n < 0)) >> 1;
        for (k = 1; k < nm1d2; k++) {
          kd = (double)k * dx;
          uu->data[k] = anew + kd;
          uu->data[n - k] = apnd - kd;
        }

        if (nm1d2 << 1 == n) {
          uu->data[nm1d2] = (anew + apnd) / 2.0;
        } else {
          kd = (double)nm1d2 * dx;
          uu->data[nm1d2] = anew + kd;
          uu->data[nm1d2 + 1] = apnd - kd;
        }
      }
    }

    t = dy / 2.0;
    kd = dy / 2.0 + 0.5;
    if (rtIsNaN(kd) || rtIsNaN(dy)) {
      n = 0;
      anew = rtNaN;
      apnd = 480.5;
    } else if (dy == 0.0) {
      n = -1;
      anew = t + 0.5;
      apnd = 480.5;
    } else if (rtIsInf(kd)) {
      n = 0;
      anew = rtNaN;
      apnd = 480.5;
    } else if (rtIsInf(dy)) {
      n = 0;
      anew = t + 0.5;
      apnd = 480.5;
    } else {
      anew = t + 0.5;
      ndbl = floor((480.5 - (t + 0.5)) / dy + 0.5);
      apnd = (t + 0.5) + ndbl * dy;
      if (dy > 0.0) {
        cdiff = apnd - 480.5;
      } else {
        cdiff = 480.5 - apnd;
      }

      kd = fabs(t + 0.5);
      if (kd >= 480.5) {
      } else {
        kd = 480.5;
      }

      if (fabs(cdiff) < 4.4408920985006262E-16 * kd) {
        ndbl++;
        apnd = 480.5;
      } else if (cdiff > 0.0) {
        apnd = (t + 0.5) + (ndbl - 1.0) * dy;
      } else {
        ndbl++;
      }

      if (ndbl >= 0.0) {
        n = (int)ndbl - 1;
      } else {
        n = -1;
      }
    }

    i2 = vv->size[0] * vv->size[1];
    vv->size[0] = 1;
    vv->size[1] = n + 1;
    emxEnsureCapacity((emxArray__common *)vv, i2, (int)sizeof(double));
    if (n + 1 > 0) {
      vv->data[0] = anew;
      if (n + 1 > 1) {
        vv->data[n] = apnd;
        nm1d2 = (n + (n < 0)) >> 1;
        for (k = 1; k < nm1d2; k++) {
          kd = (double)k * dy;
          vv->data[k] = anew + kd;
          vv->data[n - k] = apnd - kd;
        }

        if (nm1d2 << 1 == n) {
          vv->data[nm1d2] = (anew + apnd) / 2.0;
        } else {
          kd = (double)nm1d2 * dy;
          vv->data[nm1d2] = anew + kd;
          vv->data[nm1d2 + 1] = apnd - kd;
        }
      }
    }
  } else {
    for (i2 = 0; i2 < 3; i2++) {
      b_method[i2] = (method[i2] == cv2[i2]);
    }

    guard1 = false;
    if (all(b_method)) {
      guard1 = true;
    } else {
      for (i2 = 0; i2 < 3; i2++) {
        b_method[i2] = (method[i2] == cv0[i2]);
      }

      if (all(b_method)) {
        guard1 = true;
      }
    }

    if (guard1) {
      t = 639.0 / (bsize[1] - 1.0);
      if (rtIsNaN(t)) {
        n = 0;
        anew = rtNaN;
        apnd = 640.0;
      } else if ((t == 0.0) || (t < 0.0)) {
        n = -1;
        anew = 1.0;
        apnd = 640.0;
      } else if (rtIsInf(t)) {
        n = 0;
        anew = 1.0;
        apnd = 640.0;
      } else {
        anew = 1.0;
        ndbl = floor(639.0 / t + 0.5);
        apnd = 1.0 + ndbl * t;
        if (t > 0.0) {
          cdiff = apnd - 640.0;
        } else {
          cdiff = 640.0 - apnd;
        }

        if (fabs(cdiff) < 2.8421709430404007E-13) {
          ndbl++;
          apnd = 640.0;
        } else if (cdiff > 0.0) {
          apnd = 1.0 + (ndbl - 1.0) * t;
        } else {
          ndbl++;
        }

        if (ndbl >= 0.0) {
          n = (int)ndbl - 1;
        } else {
          n = -1;
        }
      }

      i2 = uu->size[0] * uu->size[1];
      uu->size[0] = 1;
      uu->size[1] = n + 1;
      emxEnsureCapacity((emxArray__common *)uu, i2, (int)sizeof(double));
      if (n + 1 > 0) {
        uu->data[0] = anew;
        if (n + 1 > 1) {
          uu->data[n] = apnd;
          nm1d2 = (n + (n < 0)) >> 1;
          for (k = 1; k < nm1d2; k++) {
            kd = (double)k * t;
            uu->data[k] = anew + kd;
            uu->data[n - k] = apnd - kd;
          }

          if (nm1d2 << 1 == n) {
            uu->data[nm1d2] = (anew + apnd) / 2.0;
          } else {
            kd = (double)nm1d2 * t;
            uu->data[nm1d2] = anew + kd;
            uu->data[nm1d2 + 1] = apnd - kd;
          }
        }
      }

      t = 479.0 / (bsize[0] - 1.0);
      if (rtIsNaN(t)) {
        n = 0;
        anew = rtNaN;
        apnd = 480.0;
      } else if ((t == 0.0) || (t < 0.0)) {
        n = -1;
        anew = 1.0;
        apnd = 480.0;
      } else if (rtIsInf(t)) {
        n = 0;
        anew = 1.0;
        apnd = 480.0;
      } else {
        anew = 1.0;
        ndbl = floor(479.0 / t + 0.5);
        apnd = 1.0 + ndbl * t;
        if (t > 0.0) {
          cdiff = apnd - 480.0;
        } else {
          cdiff = 480.0 - apnd;
        }

        if (fabs(cdiff) < 2.1316282072803006E-13) {
          ndbl++;
          apnd = 480.0;
        } else if (cdiff > 0.0) {
          apnd = 1.0 + (ndbl - 1.0) * t;
        } else {
          ndbl++;
        }

        if (ndbl >= 0.0) {
          n = (int)ndbl - 1;
        } else {
          n = -1;
        }
      }

      i2 = vv->size[0] * vv->size[1];
      vv->size[0] = 1;
      vv->size[1] = n + 1;
      emxEnsureCapacity((emxArray__common *)vv, i2, (int)sizeof(double));
      if (n + 1 > 0) {
        vv->data[0] = anew;
        if (n + 1 > 1) {
          vv->data[n] = apnd;
          nm1d2 = (n + (n < 0)) >> 1;
          for (k = 1; k < nm1d2; k++) {
            kd = (double)k * t;
            vv->data[k] = anew + kd;
            vv->data[n - k] = apnd - kd;
          }

          if (nm1d2 << 1 == n) {
            vv->data[nm1d2] = (anew + apnd) / 2.0;
          } else {
            kd = (double)nm1d2 * t;
            vv->data[nm1d2] = anew + kd;
            vv->data[nm1d2 + 1] = apnd - kd;
          }
        }
      }
    }
  }

  /*  */
  /*  Interpolate in blocks */
  /*  */
  /* blk = bestblk([nv nu]); */
  /*  LDL added */
  b_vv[0] = (unsigned int)vv->size[1];
  b_vv[1] = (unsigned int)uu->size[1];
  for (i2 = 0; i2 < 2; i2++) {
    dv2[i2] = 64.0;
  }

  rdivide(b_vv, dv2, bsize);
  b_floor(bsize);
  c_vv[0] = (unsigned int)vv->size[1];
  c_vv[1] = (unsigned int)uu->size[1];
  for (i2 = 0; i2 < 2; i2++) {
    nrem[i2] = (double)c_vv[i2] - bsize[i2] * 64.0;
  }

  emxInit_real_T(&rows, 2);
  mblocks = bsize[0];
  nblocks = bsize[1];
  i2 = rows->size[0] * rows->size[1];
  rows->size[0] = 1;
  rows->size[1] = 64;
  emxEnsureCapacity((emxArray__common *)rows, i2, (int)sizeof(double));
  for (i2 = 0; i2 < 64; i2++) {
    rows->data[i2] = 1.0 + (double)i2;
  }

  nm1d2 = vv->size[1];
  i2 = b->size[0] * b->size[1];
  b->size[0] = nm1d2;
  emxEnsureCapacity((emxArray__common *)b, i2, (int)sizeof(double));
  nm1d2 = uu->size[1];
  i2 = b->size[0] * b->size[1];
  b->size[1] = nm1d2;
  emxEnsureCapacity((emxArray__common *)b, i2, (int)sizeof(double));
  nm1d2 = vv->size[1] * uu->size[1];
  for (i2 = 0; i2 < nm1d2; i2++) {
    b->data[i2] = 0.0;
  }

  i = 0;
  emxInit_real_T(&cols, 2);
  emxInit_real_T(&v, 2);
  emxInit_int32_T(&r0, 1);
  emxInit_int32_T(&r1, 1);
  emxInit_real_T(&r2, 2);
  emxInit_real_T(&minval, 2);
  emxInit_real_T(&b_uu, 2);
  emxInit_real_T(&d_vv, 2);
  while (i <= (int)(mblocks + 1.0) - 1) {
    if (i == mblocks) {
      if (rtIsNaN(nrem[0])) {
        n = 0;
        anew = rtNaN;
        apnd = nrem[0];
      } else if (nrem[0] < 1.0) {
        n = -1;
        anew = 1.0;
        apnd = nrem[0];
      } else if (rtIsInf(nrem[0])) {
        n = 0;
        anew = rtNaN;
        apnd = nrem[0];
      } else {
        anew = 1.0;
        ndbl = floor((nrem[0] - 1.0) + 0.5);
        apnd = 1.0 + ndbl;
        cdiff = (1.0 + ndbl) - nrem[0];
        if (fabs(cdiff) < 4.4408920985006262E-16 * nrem[0]) {
          ndbl++;
          apnd = nrem[0];
        } else if (cdiff > 0.0) {
          apnd = 1.0 + (ndbl - 1.0);
        } else {
          ndbl++;
        }

        n = (int)ndbl - 1;
      }

      i2 = rows->size[0] * rows->size[1];
      rows->size[0] = 1;
      rows->size[1] = n + 1;
      emxEnsureCapacity((emxArray__common *)rows, i2, (int)sizeof(double));
      if (n + 1 > 0) {
        rows->data[0] = anew;
        if (n + 1 > 1) {
          rows->data[n] = apnd;
          nm1d2 = (n + (n < 0)) >> 1;
          for (k = 1; k < nm1d2; k++) {
            rows->data[k] = anew + (double)k;
            rows->data[n - k] = apnd - (double)k;
          }

          if (nm1d2 << 1 == n) {
            rows->data[nm1d2] = (anew + apnd) / 2.0;
          } else {
            rows->data[nm1d2] = anew + (double)nm1d2;
            rows->data[nm1d2 + 1] = apnd - (double)nm1d2;
          }
        }
      }
    }

    i2 = cols->size[0] * cols->size[1];
    cols->size[0] = 1;
    cols->size[1] = 64;
    emxEnsureCapacity((emxArray__common *)cols, i2, (int)sizeof(double));
    for (i2 = 0; i2 < 64; i2++) {
      cols->data[i2] = 1.0 + (double)i2;
    }

    for (j = 0; j < (int)(nblocks + 1.0); j++) {
      if (j == nblocks) {
        if (rtIsNaN(nrem[1])) {
          n = 0;
          anew = rtNaN;
          apnd = nrem[1];
        } else if (nrem[1] < 1.0) {
          n = -1;
          anew = 1.0;
          apnd = nrem[1];
        } else if (rtIsInf(nrem[1])) {
          n = 0;
          anew = rtNaN;
          apnd = nrem[1];
        } else {
          anew = 1.0;
          ndbl = floor((nrem[1] - 1.0) + 0.5);
          apnd = 1.0 + ndbl;
          cdiff = (1.0 + ndbl) - nrem[1];
          if (fabs(cdiff) < 4.4408920985006262E-16 * nrem[1]) {
            ndbl++;
            apnd = nrem[1];
          } else if (cdiff > 0.0) {
            apnd = 1.0 + (ndbl - 1.0);
          } else {
            ndbl++;
          }

          n = (int)ndbl - 1;
        }

        i2 = cols->size[0] * cols->size[1];
        cols->size[0] = 1;
        cols->size[1] = n + 1;
        emxEnsureCapacity((emxArray__common *)cols, i2, (int)sizeof(double));
        if (n + 1 > 0) {
          cols->data[0] = anew;
          if (n + 1 > 1) {
            cols->data[n] = apnd;
            nm1d2 = (n + (n < 0)) >> 1;
            for (k = 1; k < nm1d2; k++) {
              cols->data[k] = anew + (double)k;
              cols->data[n - k] = apnd - (double)k;
            }

            if (nm1d2 << 1 == n) {
              cols->data[nm1d2] = (anew + apnd) / 2.0;
            } else {
              cols->data[nm1d2] = anew + (double)nm1d2;
              cols->data[nm1d2 + 1] = apnd - (double)nm1d2;
            }
          }
        }
      }

      if ((!(rows->size[1] == 0)) && (!(cols->size[1] == 0))) {
        i2 = b_uu->size[0] * b_uu->size[1];
        b_uu->size[0] = 1;
        b_uu->size[1] = cols->size[1];
        emxEnsureCapacity((emxArray__common *)b_uu, i2, (int)sizeof(double));
        kd = (double)j * 64.0;
        nm1d2 = cols->size[0] * cols->size[1];
        for (i2 = 0; i2 < nm1d2; i2++) {
          b_uu->data[i2] = uu->data[(int)(kd + cols->data[i2]) - 1];
        }

        i2 = d_vv->size[0] * d_vv->size[1];
        d_vv->size[0] = 1;
        d_vv->size[1] = rows->size[1];
        emxEnsureCapacity((emxArray__common *)d_vv, i2, (int)sizeof(double));
        kd = (double)i * 64.0;
        nm1d2 = rows->size[0] * rows->size[1];
        for (i2 = 0; i2 < nm1d2; i2++) {
          d_vv->data[i2] = vv->data[(int)(kd + rows->data[i2]) - 1];
        }

        meshgrid(b_uu, d_vv, minval, v);

        /*  Interpolate points */
        /* if method(1) == 'n', % Nearest neighbor interpolation */
        /* b(i*mb+rows,j*nb+cols) = interp2(a,u,v,'*nearest'); */
        /* elseif all(method == 'bil'), % Bilinear interpolation */
        /* b(i*mb+rows,j*nb+cols) = interp2(a,u,v,'*linear'); */
        /* elseif all(method == 'bic'), % Bicubic interpolation */
        kd = (double)i * 64.0;
        i2 = r0->size[0];
        r0->size[0] = rows->size[1];
        emxEnsureCapacity((emxArray__common *)r0, i2, (int)sizeof(int));
        nm1d2 = rows->size[1];
        for (i2 = 0; i2 < nm1d2; i2++) {
          r0->data[i2] = (int)(kd + rows->data[rows->size[0] * i2]) - 1;
        }

        kd = (double)j * 64.0;
        i2 = r1->size[0];
        r1->size[0] = cols->size[1];
        emxEnsureCapacity((emxArray__common *)r1, i2, (int)sizeof(int));
        nm1d2 = cols->size[1];
        for (i2 = 0; i2 < nm1d2; i2++) {
          r1->data[i2] = (int)(kd + cols->data[cols->size[0] * i2]) - 1;
        }

        for (i2 = 0; i2 < 2; i2++) {
          bsize[i2] = minval->size[i2];
        }

        i2 = r2->size[0] * r2->size[1];
        r2->size[0] = (int)bsize[0];
        r2->size[1] = (int)bsize[1];
        emxEnsureCapacity((emxArray__common *)r2, i2, (int)sizeof(double));
        memset(&VV[0], 0, 309444U * sizeof(double));
        for (nm1d2 = 0; nm1d2 < 640; nm1d2++) {
          memcpy(&VV[1 + 482 * (nm1d2 + 1)], &a[480 * nm1d2], 480U * sizeof
                 (double));
        }

        for (nm1d2 = 0; nm1d2 < 642; nm1d2++) {
          VV[482 * nm1d2] = (3.0 * VV[1 + 482 * nm1d2] - 3.0 * VV[2 + 482 *
                             nm1d2]) + VV[3 + 482 * nm1d2];
          VV[481 + 482 * nm1d2] = (3.0 * VV[480 + 482 * nm1d2] - 3.0 * VV[479 +
            482 * nm1d2]) + VV[478 + 482 * nm1d2];
        }

        for (n = 0; n < 482; n++) {
          VV[n] = (3.0 * VV[482 + n] - 3.0 * VV[964 + n]) + VV[1446 + n];
          VV[308962 + n] = (3.0 * VV[308480 + n] - 3.0 * VV[307998 + n]) + VV
            [307516 + n];
        }

        nout = minval->size[0] * minval->size[1];
        for (k = 0; k + 1 <= nout; k++) {
          if ((minval->data[k] >= 1.0) && (minval->data[k] <= 640.0) && (v->
               data[k] >= 1.0) && (v->data[k] <= 480.0)) {
            if (minval->data[k] <= 1.0) {
              nm1d2 = 1;
            } else if (639.0 >= minval->data[k]) {
              nm1d2 = (int)floor(minval->data[k]);
            } else {
              nm1d2 = 639;
            }

            if (v->data[k] <= 1.0) {
              n = 1;
            } else if (479.0 >= v->data[k]) {
              n = (int)floor(v->data[k]);
            } else {
              n = 479;
            }

            kd = minval->data[k] - (double)nm1d2;
            t = v->data[k] - (double)n;
            ndbl = ((2.0 - kd) * kd - 1.0) * kd;
            cdiff = ((VV[(n + 482 * (nm1d2 - 1)) - 1] * ndbl * (((2.0 - t) * t -
                        1.0) * t) + VV[n + 482 * (nm1d2 - 1)] * ndbl * ((3.0 * t
                        - 5.0) * t * t + 2.0)) + VV[(n + 482 * (nm1d2 - 1)) + 1]
                     * ndbl * (((4.0 - 3.0 * t) * t + 1.0) * t)) + VV[(n + 482 *
              (nm1d2 - 1)) + 2] * ndbl * ((t - 1.0) * t * t);
            ndbl = (3.0 * kd - 5.0) * kd * kd + 2.0;
            cdiff += VV[(n + 482 * nm1d2) - 1] * ndbl * (((2.0 - t) * t - 1.0) *
              t);
            cdiff += VV[n + 482 * nm1d2] * ndbl * ((3.0 * t - 5.0) * t * t + 2.0);
            cdiff += VV[(n + 482 * nm1d2) + 1] * ndbl * (((4.0 - 3.0 * t) * t +
              1.0) * t);
            cdiff += VV[(n + 482 * nm1d2) + 2] * ndbl * ((t - 1.0) * t * t);
            ndbl = ((4.0 - 3.0 * kd) * kd + 1.0) * kd;
            cdiff += VV[(n + 482 * (nm1d2 + 1)) - 1] * ndbl * (((2.0 - t) * t -
              1.0) * t);
            cdiff += VV[n + 482 * (nm1d2 + 1)] * ndbl * ((3.0 * t - 5.0) * t * t
              + 2.0);
            cdiff += VV[(n + 482 * (nm1d2 + 1)) + 1] * ndbl * (((4.0 - 3.0 * t) *
              t + 1.0) * t);
            cdiff += VV[(n + 482 * (nm1d2 + 1)) + 2] * ndbl * ((t - 1.0) * t * t);
            ndbl = (kd - 1.0) * kd * kd;
            cdiff += VV[(n + 482 * (nm1d2 + 2)) - 1] * ndbl * (((2.0 - t) * t -
              1.0) * t);
            cdiff += VV[n + 482 * (nm1d2 + 2)] * ndbl * ((3.0 * t - 5.0) * t * t
              + 2.0);
            cdiff += VV[(n + 482 * (nm1d2 + 2)) + 1] * ndbl * (((4.0 - 3.0 * t) *
              t + 1.0) * t);
            cdiff += VV[(n + 482 * (nm1d2 + 2)) + 2] * ndbl * ((t - 1.0) * t * t);
            r2->data[k] = cdiff / 4.0;
          } else {
            r2->data[k] = rtNaN;
          }
        }

        nm1d2 = r2->size[1];
        for (i2 = 0; i2 < nm1d2; i2++) {
          n = r2->size[0];
          for (nout = 0; nout < n; nout++) {
            b->data[r0->data[nout] + b->size[0] * r1->data[i2]] = r2->data[nout
              + r2->size[0] * i2];
          }
        }

        /* end */
      }
    }

    i++;
  }

  emxFree_real_T(&d_vv);
  emxFree_real_T(&b_uu);
  emxFree_real_T(&r2);
  emxFree_int32_T(&r1);
  emxFree_int32_T(&r0);
  emxFree_real_T(&v);
  emxFree_real_T(&cols);
  emxFree_real_T(&rows);
  emxFree_real_T(&vv);
  emxFree_real_T(&uu);

  /* %% LDL if isgray(A)   % This should always be true */
  for (i2 = 0; i2 < 2; i2++) {
    bsize[i2] = b->size[i2];
  }

  i2 = minval->size[0] * minval->size[1];
  minval->size[0] = (int)bsize[0];
  minval->size[1] = (int)bsize[1];
  emxEnsureCapacity((emxArray__common *)minval, i2, (int)sizeof(double));
  i2 = (int)bsize[0] * (int)bsize[1];
  for (k = 0; k + 1 <= i2; k++) {
    if (b->data[k] <= 1.0) {
      kd = b->data[k];
    } else {
      kd = 1.0;
    }

    minval->data[k] = kd;
  }

  for (i2 = 0; i2 < 2; i2++) {
    bsize[i2] = minval->size[i2];
  }

  i2 = b->size[0] * b->size[1];
  b->size[0] = (int)bsize[0];
  b->size[1] = (int)bsize[1];
  emxEnsureCapacity((emxArray__common *)b, i2, (int)sizeof(double));
  i2 = (int)bsize[0] * (int)bsize[1];
  for (k = 0; k + 1 <= i2; k++) {
    if ((0.0 >= minval->data[k]) || rtIsNaN(minval->data[k])) {
      kd = 0.0;
    } else {
      kd = minval->data[k];
    }

    b->data[k] = kd;
  }

  emxFree_real_T(&minval);
}

static double rt_roundd_snf(double u)
{
  double y;
  if (fabs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = floor(u + 0.5);
    } else if (u > -0.5) {
      y = u * 0.0;
    } else {
      y = ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

void emxEnsureCapacity_LDL (emxArray__common *emxArray, int oldNumel, int elementSize,
	unsigned char *buf )
{
	int newNumel;
	int i;
#ifdef CUDA_OBS 
	void *newData;
#endif
	for (i = 0; i < emxArray->numDimensions; i++) {
		newNumel *= emxArray->size[i];
	}

	if (newNumel > emxArray->allocatedSize) {
		i = emxArray->allocatedSize;
		if (i < 16) {
			i = 16; 
		}   

		while (i < newNumel) {
			i <<= 1;
		}   

#ifdef CUDA_OBS 
		newData = calloc((unsigned int)i, (unsigned int)elementSize);
		if (emxArray->data != NULL) {
			memcpy(newData, emxArray->data, (unsigned int)(elementSize * oldNumel));
		if (emxArray->canFreeData) {
			free(emxArray->data);
		}   
#endif 

		printf("emxEnsureCapacity_LDL: allocatedSize %d \n", i ) ;

		emxArray->data = buf ;
		emxArray->allocatedSize = i;
		emxArray->canFreeData = false ;
	}
}

void Nimresize(const unsigned char A[921600], const double m[2],
               emxArray_uint8_T *rout, unsigned char *obuf )
{
  static double b_A[921600];
  int i0;
  emxArray_real_T *r;
  emxArray_real_T *g;
  emxArray_real_T *b;
  unsigned int varargin_1[3];
  int loop_ub;
  int b_r;
  int i1;
  double d0;
  unsigned char u0;

  /* IMRESIZE Resize image. */
  /*  LDL : 744x990 */
  /*    B = IMRESIZE(A,M,'method') returns an image matrix that is  */
  /*    M times larger (or smaller) than the image A.  The image B */
  /*    is computed by interpolating using the method in the string */
  /*    'method'.  Possible methods are 'nearest' (nearest neighbor), */
  /*    'bilinear' (binlinear interpolation), or 'bicubic' (bicubic  */
  /*    interpolation). B = IMRESIZE(A,M) uses 'nearest' as the  */
  /*    default interpolation scheme. */
  /*  */
  /*    B = IMRESIZE(A,[MROWS NCOLS],'method') returns a matrix of  */
  /*    size MROWS-by-NCOLS. */
  /*  */
  /*    RGB1 = IMRESIZE(RGB,...) resizes the RGB truecolor image  */
  /*    stored in the 3-D array RGB, and returns a 3-D array (RGB1). */
  /*  */
  /*    When the image size is being reduced, IMRESIZE lowpass filters */
  /*    the image before interpolating to avoid aliasing. By default, */
  /*    this filter is designed using FIR1, but can be specified  */
  /*    using IMRESIZE(...,'method',H).  The default filter is 11-by-11. */
  /*    IMRESIZE(...,'method',N) uses an N-by-N filter. */
  /*    IMRESIZE(...,'method',0) turns off the filtering. */
  /*    Unless a filter H is specified, IMRESIZE will not filter */
  /*    when 'nearest' is used. */
  /*     */
  /*    See also IMZOOM, FIR1, INTERP2. */
  /*    Grandfathered Syntaxes: */
  /*  */
  /*    [R1,G1,B1] = IMRESIZE(R,G,B,M,'method') or  */
  /*    [R1,G1,B1] = IMRESIZE(R,G,B,[MROWS NCOLS],'method') resizes */
  /*    the RGB image in the matrices R,G,B.  'bilinear' is the */
  /*    default interpolation method. */
  /*    Clay M. Thompson 7-7-92 */
  /*    Copyright (c) 1992 by The MathWorks, Inc. */
  /*    $Revision: 5.4 $  $Date: 1996/10/16 20:33:27 $ */
  /* [A,m,method,classIn,h] = parse_inputs(varargin{:}); */
  for (i0 = 0; i0 < 921600; i0++) {
    b_A[i0] = (double)A[i0] / 255.0;
  }

  emxInit_real_T(&r, 2);
  emxInit_real_T(&g, 2);
  emxInit_real_T(&b, 2);

  /*  Determine if input includes a 3-D array */
  /* if threeD, */
  resizeImage(*(double (*)[307200])&b_A[0], m, r);
  resizeImage(*(double (*)[307200])&b_A[307200], m, g);
  resizeImage(*(double (*)[307200])&b_A[614400], m, b);

  /* if nargout==0,  */
  /*  LDL imshow(r,g,b); */
  /* return; */
  /* elseif nargout==1, */
  /* if strcmp(classIn,'uint8'); */
  for (i0 = 0; i0 < 2; i0++) {
    varargin_1[i0] = (unsigned int)r->size[i0];
  }

  i0 = rout->size[0] * rout->size[1] * rout->size[2];
  rout->size[0] = (int)varargin_1[0];

  printf("first size %d %d %d i0 %d \n", rout->size[0], rout->size[1], rout->size[2], i0 ) ;
  emxEnsureCapacity((emxArray__common *)rout, i0, (int)sizeof(unsigned char));
  i0 = rout->size[0] * rout->size[1] * rout->size[2];
  rout->size[1] = (int)varargin_1[1];
  rout->size[2] = 3;
  // emxEnsureCapacity((emxArray__common *)rout, i0, (int)sizeof(unsigned char));
  printf("first size %d %d %d i0 %d \n", rout->size[0], rout->size[1], rout->size[2], i0 ) ;
  emxEnsureCapacity_LDL((emxArray__common *)rout, i0, (int)sizeof(unsigned char), obuf);
  loop_ub = (int)varargin_1[0] * (int)varargin_1[1] * 3;
  for (i0 = 0; i0 < loop_ub; i0++) {
    rout->data[i0] = 0;
  }

  i0 = r->size[0] * r->size[1];
  emxEnsureCapacity((emxArray__common *)r, i0, (int)sizeof(double));
  loop_ub = r->size[0];
  b_r = r->size[1];
  loop_ub *= b_r;
  for (i0 = 0; i0 < loop_ub; i0++) {
    r->data[i0] *= 255.0;
  }

  i0 = r->size[0] * r->size[1];
  for (loop_ub = 0; loop_ub < i0; loop_ub++) {
    r->data[loop_ub] = rt_roundd_snf(r->data[loop_ub]);
  }

  loop_ub = r->size[1];
  for (i0 = 0; i0 < loop_ub; i0++) {
    b_r = r->size[0];
    for (i1 = 0; i1 < b_r; i1++) {
      d0 = rt_roundd_snf(r->data[i1 + r->size[0] * i0]);
      if (d0 < 256.0) {
        if (d0 >= 0.0) {
          u0 = (unsigned char)d0;
        } else {
          u0 = 0;
        }
      } else if (d0 >= 256.0) {
        u0 = MAX_uint8_T;
      } else {
        u0 = 0;
      }

      rout->data[i1 + rout->size[0] * i0] = u0;
    }
  }

  emxFree_real_T(&r);
  i0 = g->size[0] * g->size[1];
  emxEnsureCapacity((emxArray__common *)g, i0, (int)sizeof(double));
  loop_ub = g->size[0];
  b_r = g->size[1];
  loop_ub *= b_r;
  for (i0 = 0; i0 < loop_ub; i0++) {
    g->data[i0] *= 255.0;
  }

  i0 = g->size[0] * g->size[1];
  for (loop_ub = 0; loop_ub < i0; loop_ub++) {
    g->data[loop_ub] = rt_roundd_snf(g->data[loop_ub]);
  }

  loop_ub = g->size[1];
  for (i0 = 0; i0 < loop_ub; i0++) {
    b_r = g->size[0];
    for (i1 = 0; i1 < b_r; i1++) {
      d0 = rt_roundd_snf(g->data[i1 + g->size[0] * i0]);
      if (d0 < 256.0) {
        if (d0 >= 0.0) {
          u0 = (unsigned char)d0;
        } else {
          u0 = 0;
        }
      } else if (d0 >= 256.0) {
        u0 = MAX_uint8_T;
      } else {
        u0 = 0;
      }

      rout->data[(i1 + rout->size[0] * i0) + rout->size[0] * rout->size[1]] = u0;
    }
  }

  emxFree_real_T(&g);
  i0 = b->size[0] * b->size[1];
  emxEnsureCapacity((emxArray__common *)b, i0, (int)sizeof(double));
  loop_ub = b->size[0];
  b_r = b->size[1];
  loop_ub *= b_r;
  for (i0 = 0; i0 < loop_ub; i0++) {
    b->data[i0] *= 255.0;
  }

  i0 = b->size[0] * b->size[1];
  for (loop_ub = 0; loop_ub < i0; loop_ub++) {
    b->data[loop_ub] = rt_roundd_snf(b->data[loop_ub]);
  }

  loop_ub = b->size[1];
  for (i0 = 0; i0 < loop_ub; i0++) {
    b_r = b->size[0];
    for (i1 = 0; i1 < b_r; i1++) {
      d0 = rt_roundd_snf(b->data[i1 + b->size[0] * i0]);
      if (d0 < 256.0) {
        if (d0 >= 0.0) {
          u0 = (unsigned char)d0;
        } else {
          u0 = 0;
        }
      } else if (d0 >= 256.0) {
        u0 = MAX_uint8_T;
      } else {
        u0 = 0;
      }

      rout->data[(i1 + rout->size[0] * i0) + (rout->size[0] * rout->size[1] << 1)]
        = u0;
    }
  }

  emxFree_real_T(&b);
}

/* End of code generation (Nimresize.c) */
