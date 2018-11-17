#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

#include "RndCState.h"
#include "RndC_ifc.h"

void printd(const char *fmt,
            size_t cnt,
            const double *data)
{
    size_t k;

    for (k=0; k<cnt; k++) {
        printf(fmt, data[k]);
        printf("%s", (k%10==9)?"\n":" ");
    }
    printf("%s", (k%10 != 0)?"\n\n":"\n"); 
}

void printi(const char *fmt,
            size_t cnt,
            const RndC_uint32 *data)
{
    size_t k;

    for (k=0; k<cnt; k++) {
        printf(fmt, (unsigned long)data[k]);
        printf("%s", (k%10==9)?"\n":" ");
    }
    printf("%s", (k%10 != 0)?"\n\n":"\n"); 
    fflush(stdout);
}

int main(void)
{
    RndC_uint32 seed = 0;
    unsigned long arg1, arg2;
    int n_args;
    size_t cnt=0, imx=0;
    RndC_uint32 imax=0;
    double *dout;
    RndC_uint32 *iout;
    char line[256];
    char cmnd;
    RndCState state;
    int cont = 1;
    const char *i_fmt = (sizeof(RndC_uint32)==sizeof(unsigned int))? "%6u": "%6lu";

    init_RndC(&state, seed);

    while(cont) {
        printf("Enter one of the following commdands\n"
               "\te - exit\n"
               "\tf <cnt> - Run rand (uniform distribution between 0 to 1)\n"
               "\ti <imax> <cnt> - Run randi (random integer between 1 to imax)\n"
               "\tn <cnt> - Run randn (stndard normal distribution)\n"
               "\tp <imax> [<cnt>] - Run randperm (get <cnt> different random numbers\n"
               "\t                   between 1 and imax. If cnt is omitted, get a permutation\n"
               "\t                   of length imax\n"
               "\ts <seed> - set the seed to seed (initial seed is zero\n"
               );

        if(fgets(line, sizeof(line), stdin) == NULL) {
            fprintf(stderr, "**** Error reading next command ****\n");
            exit(EXIT_FAILURE);
        }
        n_args = sscanf(line, " %1s %lu %lu", &cmnd, &arg1, &arg2);
        switch (n_args) {
        case 3:
            imax = (RndC_uint32) arg1;
            imx = (size_t) arg1;
            cnt  = (size_t) arg2;
            break;
        case 2:
            cnt  = (size_t) arg1;
            imx = cnt;
            break;
        }
        if(n_args <= 0)
            break;


        switch(cmnd) {
        case 'e': cont=0; break;
        case 'f':
            if(n_args <2)
                break;
            dout = ( double * )malloc(sizeof(*dout)*cnt);
            rand_RndC(&state, cnt, dout);
            printd("%6.4f", cnt, dout);
            free(dout);
            break;
        case 'i':
             if(n_args <3)
                break;
            iout = ( RndC_uint32 * )malloc(sizeof(*iout)*cnt);
            randi_RndC(&state, imax, cnt, iout);
            printi(i_fmt, cnt, iout);
            free(iout);
            break;
        case 'n':
            if(n_args <2)
                break;
            dout = ( double * )malloc(sizeof(*dout)*cnt);
            randn_RndC(&state, cnt, dout);
            printd("%6.3f", cnt, dout);
            free(dout);
            break;
        case 'p':
            if(n_args < 2)
                break;
            iout = ( RndC_uint32 * )malloc(sizeof(*iout)*cnt);
	    if(n_args == 3)
                randperm_RndC(&state, imx, cnt, iout);
            else
                randperm1_RndC(&state, imx, iout);
            printi(i_fmt, cnt, iout);
            free(iout);
            break;
        case 's':
            if(n_args < 2)
                break;
            seed = arg1;
            init_RndC(&state, seed);
            printf("seed set to %lu\n", (unsigned long)seed);
            break;
        default:
            fprintf(stderr, "Unexpected command\n");
            break;
        }
    }
    return EXIT_SUCCESS;
}
