# include path
INCLUDE_DIRS:=.
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

TST_C_SRCS:= AcceptTCPConnection.c CreateTCPServerSocket.c DieWithError.c HandleTCPClient.c

CS_TCP_SOCKET_LIB:= $(TST_LIB_DIR)/libcs_tcp_socket$(LIB_EXT)
$(CS_TCP_SOCKET_LIB): $(patsubst %.c,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(TST_C_SRCS))

TST_LIB_FILES:= $(CS_TCP_SOCKET_LIB)

include $(ROOT_DIR)/util/rules.mak

# all : AcceptTCPConnection.o CreateTCPServerSocket.o DieWithError.o HandleTCPClient.o
# 
# %.o : %.c
# 	g++ -c $<
# 
# clean:
# 	rm -f *.o

