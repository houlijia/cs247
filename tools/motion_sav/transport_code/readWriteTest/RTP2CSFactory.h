#ifndef RTP2CS_FACTORY_H
#define RTP2CS_FACTORY_H

#include <map>
#include <iostream>

#include "RtpPacket.h"
#include "CodeElement.h"
#include "common_ljw.h"
#include "CircularQueue.h"
#include "readUInt.h"
#include "writeUInt.h"
#include "BufferSender.h"
#include "stdlib.h"

using std::map;
using std::cerr;
using std::endl;

 static uint32 numberOfDropPkt = 0;
/**
 * @brief This is the struct representing a sub-block. It contains
 * the buffer, the data length, and the number of integers inside it 
 */
struct MeasurementsS 
{
 uint32 len; /**< The length of the data part in bytes.  */
 uint32 nInteger; /**< The number of integers inside this sub-block**/
 unsigned char* buffer; /**< The data that comes from one RTP packet, should use
                         * malloc for allocation. */
 
 /**
  * @brief The constructor will use malloc to allocate a space of len0 bytes.
  * Then, it will copy len0 bytes from the buffer0 to the inside buffer.
  * then, it counts the number of integers inside it. 
  */ 
 MeasurementsS(uint32 len0, char* const buffer0) : len(len0) 
 {
  buffer = (unsigned char*) malloc(len0);
  memcpy(buffer, buffer0, len0);
  updateNIntegers();
 };
 
 ~MeasurementsS()
 {
  free(buffer);
 }
 
 /**
  * count the number of integers, and update the member
  */ 
 void updateNIntegers()
 {
  this->nInteger = countNumbers(this->buffer, this->len);
 }
 
};
typedef struct MeasurementsS Measurements;  


typedef  map<uint32, Measurements*> MVector;

/**
 * This represents the first three global meta CE
 */
struct GlobalMetaStruct
{
 bool* metaFlags; /**< The flag indicating whether the meta data CodeElement is received or not*/
 CodeElement* HeadCE; /**< The place to store the meta data CodeElements */
 BufferSender* sender; /**< The sender to write to file/TCP*/
 bool isSent; /**< Indicating whether it has been sent by the sender */
 const uint8 NUMBER_OF_GLOBAL; /** < A constant - The number global meta CEs, 
                                Now, it is 3. */
 
 GlobalMetaStruct(BufferSender* sender0) : sender(sender0), 
 NUMBER_OF_GLOBAL(3)
 {
  metaFlags = new bool[NUMBER_OF_GLOBAL]; 
  HeadCE = new CodeElement[NUMBER_OF_GLOBAL];
  for(int i=0; i<NUMBER_OF_GLOBAL; i++)
   metaFlags[i] = false;
  isSent = false;
 }
 ~GlobalMetaStruct()
 {
  delete[] metaFlags;
  delete[] HeadCE;
 }
 
 bool
 checkOneMetaReceived (char metaIndex)
 {
  assert (metaIndex >= 0 && metaIndex < NUMBER_OF_GLOBAL);
  return metaFlags[metaIndex];
 }

 /**
  * @breif check whether all the global meta CEs are received
  */
 bool checkGlobal()
 {
  for(uint8 i=0; i < NUMBER_OF_GLOBAL; i++)
   {
    if(metaFlags[i] == false) return false;
   }
  return true;
 }

/**
 * @brief send the global meta first here, will set the isSent to true
 * The user can check that flag, do not need to send twice
 */ 
 void process(const char** errString)
 {
  uint32 bytesW = 0;
  //write the first three meta data into sender
  //do not need to check the size, since those are pretty small
  for(uint8 i=0; i<NUMBER_OF_GLOBAL; i++)
   {
    bytesW += (this)->HeadCE[i].writeToBuffer((sender->buffer + bytesW), errString);
   }
  sender->send(bytesW);
  this->isSent = true;
  
 }
 
};


typedef struct GlobalMetaStruct GlobalMeta;  
/**
 * @brief This is the data structure that represent a Block
 */
struct BlockStruct
{
 bool* metaFlags; /**< The flag indicating whether the meta data CodeElement is received or not*/
 CodeElement* HeadCE; /**< The place to store the meta data CodeElements */
 const uint8 NUMBER_OF_META; /** < This is the number of meta CE except for the global ones ,now 4*/
 map<uint32, Measurements*> dataMap; /** < The map that will store the data pieces of one block from packets, and make users easy to iterate them into sorted data to form the data part of one block*/
 //  longInt subNo; /** < The number of sub blocks we are expecting to receive*/
 // bool subNoSet; /** < The flag indicating whether the subNo field of this block is set or not*/
 bool isComplete; /**< The flag that indicating whether the metadata and data is complete*/
 longInt totalNumbers; /** < this is the total number of integers we should 
                        * expect to receive inside the block, it is set when
                        *  we check whether the data part is complete. It is 
                        * read from the meta part of CodeElement 6. */ 
 uint8 blockNumber;  ///< blockNumber of the current block
 GlobalMeta* globalMeta; ///< a pointer to the shared global meta 
 BufferSender* sender; ///< sender passed from the user

 int nbins;  ///the first number of CE 6, number  of bins
 
 
 BlockStruct(const uint8 blockNumber0, BufferSender* sender0, GlobalMeta* gm0): 
 blockNumber(blockNumber0), globalMeta(gm0), sender(sender0), NUMBER_OF_META(4),
 isComplete(false)
 {
  metaFlags = new bool[NUMBER_OF_META];
  HeadCE = new CodeElement[NUMBER_OF_META];
  for(uint8 i =0; i< NUMBER_OF_META; i++)
   metaFlags[i] = false;
 }
 
 //when you delete a BlockStruct* int the CircularQueue, make sure you set the pointer to NULL 
 ~BlockStruct()
 {
  const char* err= NULL;
  
  //first call the sender to send it before deconstruction
  process(&err);
  
  MVector::iterator iter;
  for(iter = dataMap.begin(); iter != dataMap.end(); iter++)
   {
    delete (iter->second);
    iter->second = NULL;
   }
  delete[] metaFlags;
  delete[] HeadCE;
 }

/**
 * @return 0 written to file/tcp
 *         1 dropped because the meta data is not complete
 */
 
 char process(const char** errString)
 {

  //will drop if meta is not complete
  cerr<<"processing block "<<(int)blockNumber<<endl;
  if( (this)->checkAllMetaReady() )
   {
    uint32 bytesW = 0;
    if(globalMeta->isSent == false)
     {
      globalMeta->process(errString);
     }
    //write the meta data 345 to the buffer inside the sender
    for(uint8 i=0; i<NUMBER_OF_META-1; i++) {
      bytesW += (this)->HeadCE[i].writeToBuffer((sender->buffer) + bytesW, errString);
     }
    cerr<<"3, 4, 5 len:"<<bytesW<<endl;

   int endP;
    char temp[10];
    const char* dataStart = HeadCE[NUMBER_OF_META-1].getData();
    this->nbins = readOnlyOneUInt(dataStart , 0, &endP, errString);
    longInt fillValue = this->nbins +2;
    int fillBytes = writeOneUIntExtern (fillValue, temp, errString);

    uint8 pos = findPos((const unsigned char*)(dataStart+endP +1), 0, 2, 20, errString);
    
    longInt len_s = readOnlyOneUInt(dataStart+endP+pos+2 , 0, &endP, errString);
    assert(len_s ==0);
    
    //then compose 6 and write into buffer
    longInt blockDataLen = (this)->getTotalLen();
    cerr<<"blockDataLen only ="<<blockDataLen<<endl;

    int lossBytes = this->countLoss();
    assert(lossBytes >=0);
    cerr<<"lossBytes ="<<lossBytes<< "  "<<lossBytes*fillBytes <<endl;

    uint32 metaLen = (this)->HeadCE[NUMBER_OF_META -1].getLength();
    blockDataLen += (metaLen + lossBytes * fillBytes); 
    
    cerr<<(int)blockNumber<<" blockDataLen ="<<blockDataLen<<endl;
    cerr<<"metaLen="<<metaLen<<endl;
    
    //write the key, len of 6, used a fixed number here
    //remember to change if the key of measurement CE is changed
    bytesW += writeOneUIntExtern(6, (sender->buffer +bytesW),  errString ); 
    bytesW += writeOneUIntExtern(blockDataLen, (sender->buffer +bytesW),  errString );
   //this may be not enough here if there was data packet loss 
    sender->checkBufferSize(bytesW + blockDataLen);
    //copy the meta data
    memcpy((sender->buffer +bytesW), (this)->HeadCE[NUMBER_OF_META -1].getData(), metaLen);
    
    bytesW += metaLen;
    //write all the map data into buffer
    int bb =(this)->writeDataMapToBuffer((sender->buffer +bytesW), fillValue, errString);
    cerr<<"written= "<<bb<<endl;
    bytesW +=bb;
    //we may have FileSender, or TCPSender, they both have a buffer inside
    //call the send function of the sender
    sender->send(bytesW);
    return 0;
   }
   else
{
	numberOfDropPkt++;
	return 1;
}
 }
 
/**
 * @return true is ready
 */ 
 bool checkAllMetaReady ()
 {
//  return false;
  if(!(globalMeta->checkGlobal()))
   return false;
  for(uint8 i=0; i < NUMBER_OF_META; i++)
   {
    if(metaFlags[i] == false) return false;
   }
  return true;
 }
 
 bool printAllMetaReady ()
 {
  for(uint8 ii=0; ii <NUMBER_OF_META ; ii++)
   {
    cerr<<"meta["<<ii<<"]=" << metaFlags[ii] <<endl;
   }
  return true;
 }
 ///@brief If the last several packets are lost, we can not detect it.
#if 0
 bool checkDataComplete()
 {
  uint32 current = 0;
  uint32 len =0;
  MVector::iterator iter;
  for(iter = dataMap.begin(); iter != dataMap.end(); iter++)
   {
    if(iter->first != current + len);
    return false;
    else
     {
      current = iter->first;
      len = iter->second->len;
     }
   }//for
  return true;
 }
#endif


 /**
  *@brief This function check whether the data part is complete.
  * you need the CE 6 meta part received first. or it will always return false. 
  *@return true is complete
  */
 bool checkDataComplete(const char** errString)
 {
  if(metaFlags[NUMBER_OF_META-1] == false)
   return false;
  else
   {
    int endP;
    const char* dataStart = HeadCE[NUMBER_OF_META-1].getData();
    uint8 pos = findPos((const unsigned char*)dataStart, 0, 2, 20, errString);
    //read the len_b
    this->totalNumbers = readOnlyOneUInt(dataStart + pos+1, 0, &endP, errString) -1;
    MVector::iterator iter;
    uint32 realLen =0;
    uint i =0;
    for(iter = dataMap.begin(); iter != dataMap.end(); iter++)
     {
      
      realLen += iter->second->nInteger;
      i++;
     }
    cerr<<"realLen="<<realLen<<"  totalNumbers="<<totalNumbers<<endl;
    if(realLen == totalNumbers)
     return true;
    else
     return false;
    
   }
 }
 
 /**
  * @brief This function will write the data in the dataMap to the buffer
  * The idea is 
  * @return The number of Bytes written
  */
 uint32 writeDataMapToBuffer(char* buffer, longInt fillValue, const char** errString)
 {
  uint32 bytesW = 0; 
  
  MVector::iterator iter;
  int expected =0;
int nZero = 0; 
    int nloop =0;


  for(iter = dataMap.begin(); iter != dataMap.end(); )
   { 
    nloop++;

      cerr<<"nloop out"<<nloop<<" " <<iter->first<< " "<<expected<<endl;
    if(iter->first != expected)
     {
      //fill from expected to iter->first -1 
      nZero = (iter->first) - expected;
      //if(nZero <0)
      //cerr<<"nloop "<<nloop<<" " <<iter->first<< " "<<expected<<endl;
      assert(nZero >=0);
      for(int i=0; i< nZero; i++)
       {
        bytesW += writeOneUIntExtern (fillValue, buffer+bytesW, errString);
       }
      //memset(buffer+bytesW, 0, nZero);
      //bytesW += nZero;
      expected += nZero;
     }
    else
     {
      memcpy(buffer+bytesW, iter->second->buffer, iter->second->len);
      bytesW += iter->second->len;
      expected += iter->second->nInteger;
      iter++;
     }

     //need to check whether expected == totalNumber
        }

    if(expected != this->totalNumbers)
      {
          assert(expected < this->totalNumbers);
          nZero = this->totalNumbers - expected;
          for(int i=0; i< nZero; i++)
          {
            
            bytesW += writeOneUIntExtern (fillValue, buffer+bytesW, errString);
          }
      }

  return bytesW;
  
 }

 uint32 countLoss()
 {
   MVector::iterator iter;
   uint32 lb =0;
   int nZero =0;
   int expected = 0;
   for(iter = dataMap.begin(); iter != dataMap.end(); )
   { if(iter->first != expected)
     {
       //fill from expected to iter->first -1 
       nZero = (iter->first) - expected;
       lb += nZero;
       expected += nZero;
     } 
     else
     {
       expected += iter->second->nInteger;
       iter++;
     }
   }

   if(expected != this->totalNumbers)
   {
     assert(expected < this->totalNumbers);
     nZero = this->totalNumbers - expected;
     lb += nZero;
   }
   return lb;
 }



/**
 * @brief This function adds up the number of integers inside the dataMap
 * The result is the number of integers currently received for this block
 */ 
 uint32 getTotalLen()
 {
  
  MVector::iterator iter;
  uint32 realLen = 0;
  for(iter = dataMap.begin(); iter != dataMap.end(); iter++)
   {
    realLen += iter->second->len;
   }
  return realLen;
 }
 
 
 
 
 /* @brief The function for checking whether one meta data block is received or not, can check 0 to 6
  * @return 0 not received, 1 received
  */
 bool
 checkOneMetaReceived (uint8 metaIndex)
 {
  assert (metaIndex >= 0 && metaIndex < NUMBER_OF_META);
  return metaFlags[metaIndex];
 } 
 
 bool checkComplete(const char** errString)
 {
  this->isComplete = (checkAllMetaReady() && checkDataComplete(errString));
  cerr<<checkAllMetaReady()<<" "<< checkDataComplete(errString) <<" "<<isComplete<<endl;
  return (this->isComplete);
 }
 
};

typedef struct BlockStruct Block;

class RTP2CSFactory
{
public:
 
 RTP2CSFactory (uint waitblocksbeforedrop0, BufferSender* sender0) : waitWindowSize(waitblocksbeforedrop0),    blockQP (new CircularQueue<Block*>(waitWindowSize, true)),  blockQ(*blockQP), sender(sender0), maxBlockNumber (-1), globalMeta(sender0), numberOfLate(0)
 {
  
 };
 
 ~RTP2CSFactory()
 {
  //first clean up the things in the queue, because you can not rely on 
  //the deconstructor to do that. It will clean from the array index, not the
  //order we want
  blockQP->cleanQueue();
  //cerr<<"i from clean:"<<i<<endl;
  //cerr<<"block Number that is not nULL"<<(int)( (blockQ[i])->blockNumber)<<endl;
  delete blockQP;
  blockQP = NULL;
  
  //clean sender at last
  if(sender != NULL)
   {
    delete sender;
    sender =NULL;
   }

   cout<<"total late packets dropped: "<<numberOfLate<<endl;
   cout<<"total dropped incomplete block: "<<numberOfDropPkt<<endl;
 }
 
 //blockN can be 0
 //we assume the later one is the newBlock number
 //but if it is not, the function will detect it, and return a negative number 
 int calcBlockOffset(int oldBlockN, int newBlockN)
 {
  int larger = newBlockN;
  int largerOld = oldBlockN;
  int small_window = 64;
  
  if(abs(oldBlockN -128)< small_window && abs(newBlockN -128) < small_window)
   return newBlockN - oldBlockN;
  if(newBlockN < 128)
   {
    larger = newBlockN + 256;
   }
  
  if(oldBlockN < 128)
   {
    largerOld = oldBlockN + 256;
   }
  
  return (larger - largerOld);
 }
 
 /*
  * @brief Designed so that reading from file and reading from UDP can both use this function
  * The user of this function can delete/free the packet after calling this function, since the data is copied.
  * 
  */
 void
 insertRTPPacket (RtpPacket* const rtpP, /**< The RTP Packet pointer*/
 uint32 allLen,  /**< The total length of the rtp packet, This can be achieve from UDP paylaod length, or the file zie*/
 const char** errString /**< The error string*/
 )
 {
  // dq.push_back (rtpP);
  //check the meta
  CodeElement ceS;
  uint32 keyLenSize;
  uint32 bytesR = 0;
  uint8 blockNumber =0 ;
  
  longInt dataOffset = 0;
  int realKey = 0;
  //first read the block number, to see whether this is a new block
  memcpy(&blockNumber, rtpP->getData(), 1);
  
  assert(blockNumber >=0);
  cerr<<"blockNumber =" <<(int)blockNumber<<endl;
  int aa;  
  if((aa =calcBlockOffset( maxBlockNumber, blockNumber)) >0)
   {
    maxBlockNumber = blockNumber;
   }
  
  cerr<<"block offset to max ="<<aa<<endl;
  bytesR = 1;
  int blockOffset;
  
  //if we ever cleaned all the blocks, we will only accept new block that is
  //larger than the maximum of the old one
  int currentQLen = blockQ.getLen();
  if(currentQLen ==0 && aa >0)
   {
    Block* newBlock = new Block(blockNumber, sender, &globalMeta);
    blockQ.insertToRear(newBlock, errString);
    blockOffset = 0;
   }
  else if(currentQLen ==0)
   {
    //skip this rtp
    return;
   }
  else if (currentQLen >0)
   {
    cerr<<"@@ in else"<<endl;
    blockOffset = calcBlockOffset((blockQ.getOldest())->blockNumber, blockNumber);
    
    cerr<<"blockOffset ="<<blockOffset<<endl;
    if(blockOffset <0)
     {
	numberOfLate++;
      fprintf(stderr, "too late, do not want this packet anymore: %d\n", blockNumber);
      return;
     }
    if(blockQ.isIndexInsideWindow(blockOffset))
     {

    //check whether it is NULL there, if yes, new a block
      if(blockQ[blockOffset] == NULL)
       {
        Block* newBlock = new Block(blockNumber, sender, &globalMeta);
        blockQ.insertToInsideW(newBlock, blockOffset, errString);
       }

        //if not null, just insert the data into the existing dataMap
     }
    else
     {
      //if it is outside the current window, we definitely need to new a block
      Block* newBlock = new Block(blockNumber, sender, &globalMeta);

      //this will clean the old stuff inside, and then insert
      int timeoutN = blockQ.cleanForInsertTo(newBlock, blockOffset, errString);
     // cerr<<"timeoutN" <<timeoutN<<endl;
      //assert(timeoutN ==1);
      //call aggressiveClean() here
      int nClean = aggressiveClean(errString);
      //assert(nClean ==0);
      //cerr<<"insertOff"<< blockOffset <<"  "<<nClean<<"  "<<blockQ.getBaseIndex()<<endl;
      blockOffset -= nClean + timeoutN;
      blockQ.insertToInsideW(newBlock, blockOffset , errString);
      
     }
   }
  
  keyLenSize = ceS.readKeyLenFromBuffer (rtpP->getData () + bytesR, errString);
  bytesR += keyLenSize;
  
  //if not received yet CE 0/1/2
  if (!(globalMeta.checkOneMetaReceived (ceS.getKey () )))
   {
    //need to copy the data here, since it is hard to free the packet later due to the uncertain order of packets later
    //if we only set the pointer here
    ceS.setData (rtpP->getData () + bytesR, keyLenSize);
    clone_CE (&ceS,  & (globalMeta.HeadCE[(ceS.getKey ())]),  errString);
    globalMeta.metaFlags[(ceS.getKey ())] = true;
    checkError(*errString);
   }
  
  bytesR += ceS.getLength();
 

  for(uint8 i =0; i<3; i++)
   {
  //this if reading 3
  keyLenSize = ceS.readKeyLenFromBuffer (rtpP->getData () + bytesR, errString);
  bytesR += keyLenSize;
  
  realKey = ceS.getKey () - (globalMeta.NUMBER_OF_GLOBAL); 
  if (!blockQ[blockOffset]->checkOneMetaReceived (realKey))
   {
    //need to copy the data here, since it is hard to free the packet later due to the uncertain order of packets later
    //if we only set the pointer here
    ceS.setData (rtpP->getData () + bytesR, keyLenSize);
    clone_CE (&ceS,  & ((blockQ[blockOffset])->HeadCE[realKey]),  errString);
    (blockQ[blockOffset])->metaFlags[realKey] = true;
   }
  bytesR += ceS.getLength();
   }

  
  ceS.setData(NULL, 0);
  //read the offset
  int endP;
  dataOffset= readOnlyOneUInt(rtpP->getData() + bytesR, 0, &endP, errString );
  checkError(*errString);
  
  bytesR += (endP+1);
  //read data
  uint32 dataLen = allLen - (rtpP->getHeaderLength()) - bytesR;
  Measurements* mp = new Measurements(dataLen, rtpP->getData()+bytesR);
  blockQ[blockOffset]->dataMap.insert(std::make_pair(dataOffset, mp));
  
  //check if the current block is complete
  //set the isComplete to true if it is complete
  // process the block if it is the oldest block in the queue
  //  cerr<<"@@@@@blockNumber="<<(int)blockNumber<<endl;
  if(blockQ[blockOffset]->checkComplete(errString))
   {
    checkError(*errString);
    if(blockOffset ==0)
     {
      //not only processing the block, process until you met a non complete one
      aggressiveClean(errString);
     }
   }
 }


 /**
  * Process the block until you see a NULL or incomplete one
  * @return the number of elements cleaned
  */
 int aggressiveClean(const char** errString)
 {
  int i=0;
  //write the block into file, free the block
  //if its isComplete =true, then process, and deleteEnd
 // cerr<<" in agrre"<<blockQ[0]<< "  " <<blockQ[0]->isComplete<<" base ="<<blockQ.getBaseIndex() <<
      //"rear "<<blockQ.getRearIndex() <<endl;
  while(blockQ.getLen() >0 )
   {
    if(blockQ[0] != NULL)
     {
      if(blockQ[0]->isComplete)
       {
        cerr<<"isInside" <<blockQ.getLen()<<" base ="<<blockQ.getBaseIndex()<<endl;
        blockQ.deleteEnd();
        
        cerr<<"after delete" <<blockQ.getLen()<<" base ="<<blockQ.getBaseIndex() <<
          "rear "<<blockQ.getRearIndex() <<endl;
       }
      else
       {
        break;
       }
     }
    else
     {
      
        blockQ.deleteEnd();
     }
    i++;
    //  cerr<<" after delte "<<(blockQ[0] != NULL)<< "  " <<blockQ[0]->isComplete<<endl;
   }
  return i;
 }
 
 
 
private:
 
 GlobalMeta globalMeta; ///< CE 0, 1,2 shared by all the blocks 
 //did not set the window size to const, maybe we can change it to control how we drop packet in the middle of one transmission?
 uint8 waitWindowSize; ///< the window size, until you must process it
 uint32 numberOfLate;
 
 
 int maxBlockNumber; ///< The maximum of blockNumbers seen, set to int here because I set it to -1 initially
 BufferSender* sender; ///< an abstract sender pointer, can be set to any BufferSender
 
 //used vector instead of deque, because we have many accesses based on index
 CircularQueue<Block*>* blockQP;
 CircularQueue<Block*>& blockQ;
 
 
 
};
#endif	
