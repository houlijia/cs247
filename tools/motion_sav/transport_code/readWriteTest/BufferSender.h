/* 
 * @file   BufferSender.h
 * @author: Jianwei Liu 
 *
 * Created on July 9, 2014, 3:21 PM
 */

#ifndef BUFFERSENDER_H
#define	BUFFERSENDER_H
#include "common_ljw.h"

/**
 * @class This is an abstract class for a sender. It must has a buffer to put things
 * it to. And it must have a send function.   
 * @param init_size
 */
class BufferSender
{
public:
 BufferSender (uint32 init_size){ 
  size = init_size;
  buffer = (char *) malloc(init_size);
 };
 BufferSender (const BufferSender& orig);

 /**
  * @brief The function to resize the buffer of the BufferSender
  * @param newSize -The new size user wants to reset the buffer size
  * @return 0 succ  -1 fail 1 did not change 
  */
 char checkBufferSize(uint32 newSize)
 {
  if(newSize > size)
   {
  buffer =(char*) realloc(buffer, newSize);
  if(buffer ==NULL)
   return -1;
  else
   return 0;
   }
  else
   return 1;
 }


/**
 * The send function that every child class should have
 * @param sendLen
 * @return 
 */
 virtual uint32 send(uint32 sendLen) =0;
 virtual ~BufferSender ()
 {
  if(buffer !=NULL)
  free(buffer);
 buffer = NULL;
 }
 char* buffer;
private:
 uint32 size;

};

#endif	/* BUFFERSENDER_H */

