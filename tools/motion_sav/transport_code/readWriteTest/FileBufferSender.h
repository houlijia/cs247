/* 
 * File:   FileBufferSender.h
 * Author: jliu121
 *
 * Created on July 9, 2014, 3:28 PM
 */

#ifndef FILEBUFFERSENDER_H
#define	FILEBUFFERSENDER_H
#include "BufferSender.h"
#include <iostream>
#include <stdio.h>
using std::cerr;
using std::endl;

class FileBufferSender : public BufferSender
{
public:

 /**
  * Just get a pointer from user, and store it locally. Call the constructor of 
  * BufferSender, init a buffer of size "size"
  * @param size
  * @param fp0
  */
 FileBufferSender (uint32 size, FILE* fp0): BufferSender(size), fp(fp0) {
 };

 /**
  * The send function required by BufferSender 
  * @param sendLen The number of bytes that needs to be sent
  * @return The number of bytes sent 
  */
 uint32 send( uint32 sendLen)
 {
//  cerr<<"sendLen inside"<<sendLen<<endl;
  int sent = fwrite(buffer, 1, sendLen, fp);
  return sent;
 }
// FileBufferSender (const FileBufferSender& orig);
 virtual ~FileBufferSender (){

 };
private:
 FILE* fp;

};

#endif	/* FILEBUFFERSENDER_H */

