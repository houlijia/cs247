CC_OPTS:= $(INCLUDE_OPT) -fdiagnostics-show-option -Wall -Werror
CPP_OPTS:= $(INCLUDE_OPT) -pedantic-errors -fdiagnostics-show-option -Wall -Werror\
   -ansi -std=c++0x
TST_CC_SYS_LIBS+= -lm

ifneq ($(filter kanas%,$(HOSTNAME)),)
CC_OPTS+=  -DOPENCV_3
CPP_OPTS+= -DOPENCV_3
endif
