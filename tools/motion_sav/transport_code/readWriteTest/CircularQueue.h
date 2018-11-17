/* 
 * File:   CircularQueue.h
 * Author: Jianwei Liu 
 *
 * Created on July 7, 2014, 5:23 PM
 */

#ifndef CIRCULARQUEUE_H
#define	CIRCULARQUEUE_H

#include "common_ljw.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
using std::endl;
using std::cerr;

template <class Type>
class CircularQueue {
 public:
  
  /**
   * @brief Constructor for a fixed-size circular queue, though the size can be 
   * changed by calling resize. But, most of time, you use it as a fix-sized 
   * wrap-around queue.  
   * @param size0 -The size 
   * @param isPointer -A flag indicating whether the elements stored are pointers.
   */
  CircularQueue(uint32 size0, bool isPointer) :size(size0), 
    pointersStored(isPointer) 
  {
   array = (Type*) malloc(sizeof(Type) * size);
   if(pointersStored)
    {
     for(int i=0; i<size; i++)
      array[i] = NULL;
    }
   baseIndex = -1;
   rearIndex = -1;
  }
  int calcDistToBase(int newIndex )
  {
   if(baseIndex > newIndex)
    return size-baseIndex +newIndex;
   else
    return newIndex - baseIndex;
  }
  
  
  /**
   *@brief This function insert an element to the rear of the queue. You must
   * call this function instead of the interToIndex if there is not element
   * inside it. 
   *  If there was any element there, it will be freed first
   * , and set to the new one.
   * If pointers stored, must delete/free it first, and set to NULL
   * There is an assert in the function for that!!! 
   * If instances are stored, make sure there is a proper copy constructor 
   * that does what you want.
   
   * @param t -The element to insert
   */
  void insertToRear(Type t /**< the element we are going to insert, it can be an
                            *  pointer*/,
                    const char** errString)
  {
   //uint32 realIndex= (baseIndex + offset) % size;
   //        cerr<<"baseIndex="<<baseIndex<<" offset="<<offset<<endl;
   if(rearIndex+1 != baseIndex)
    rearIndex += 1;
   else
    {
     *errString = "circular queue full, should clean the oldest element first before inserting.\n";
     return;
    }
   assert(rearIndex ==0);
   
   
   array[rearIndex] = t;
   if(baseIndex ==-1)
    baseIndex =0;
   
  }
  bool isIndexInsideWindow(int index)
  {
   return (index >=0 && index<size);
  }
  

  /**
   * @breif This function can only be called, if the index is inside the window
   * call isInsideWindow to check that. The queue can be empty, in that case, 
   * insertToRear() will be called inside. Also, need to make sure the insert point
   * of the array is NULL.  
   * @param t The element to insert
   * @param index The index you want to insert to, from baseIndex, >=0 , <size
   * @param errString
   */
  void insertToInsideW(Type t, 
                       int index,
                       const char** errString)
  {

   if(getLen()==0)
    {
     insertToRear(t, errString);
     return;
    }
   // assert(getLen() >0);
   assert(index >=0  && index <size);
   
   
   int insertIndex = (index+baseIndex) %size;
   if(pointersStored)
    assert(array[insertIndex] == NULL);
   
   array[insertIndex] = t;
   
   
   //its wrong here, need to calc distance
   if((index+1) > this->getLen()) 
    rearIndex = insertIndex;
   cerr<<"base" <<baseIndex<<" rear"<<rearIndex<<endl;
   
  }
  
  /**
   * This function will clean the old element before inserting
   * new element. Only call this function is index is larger than size-1 
   * examples: I for incomplete; C for complete
   * | 0 | 1 | 2 |
   *   I   C   C
   * if 3 comes, we will clean 0
   * if 6 comes, we will clean 0, 1, 2, and then set the baseIndex and rearIndex 
   * to -1;
   * user can then call inserToInsideW to insert the new block
   * I did not insert here, because I want user to call the aggressiveClearn outside
   * like the case of 3 comes, we clean 0 first, and then user can call aggresiveClean
   * to clean 1 and 2. After aggressive clean, we will call insertToInsideW.
   * 
   * for the case
   * |0|1 | NULL | 
   * we can not always insert to rear, because we need to leave space for some
   * blocks, e.g. we got 0, 1, 2, 
   *                     I  C  I  
   * here comes 4, then, 0, 1 will be processed, if we put 4 in the rear, the sequence
   * will be wrong, and 3 will not have space later.
   * 
   * 
   * @param t
   * @param index
   * @param errString
   * @return The number of elements cleaned
   */
  
  int cleanForInsertTo(Type t, 
                       int index,
                       const char** errString)
  {
   assert(getLen() >0);
   assert(index >=size );
   assert (isIndexInsideWindow(index) == false);
   
   //int insertIndex = (index+baseIndex) %size;
   
   //check whether we need to clean any thing
   int offIn = index - this->size;
   int oldLen = this->getLen();
   
   if(offIn > oldLen -1)
    offIn = oldLen -1;
   assert(offIn >=0);
   
   if(pointersStored)
    {
     for(int i=0; i<= offIn; i++)
      {
       if((*this)[i] != NULL)
        {
         delete ((*this)[i]);
         (*this)[i] = NULL;
        }
      }
    }
   
   if(offIn != oldLen -1)
    {
     baseIndex = (baseIndex +offIn +1) %size;
    }
   else
    {
     baseIndex = -1;
     rearIndex = -1;
    }
   cerr<<"base"<<baseIndex<<"  rear" <<rearIndex<<"offIn "<<offIn<<endl;
   return offIn +1;
  }
  
  
  
  /**
   * @brief Delete one element from the end  of the queue. 
   */ 
  void deleteEnd()
  {
   uint32 nextIndex=0;
   if(pointersStored)
    {
     if(array[baseIndex] != NULL)
      {
       delete array[baseIndex];
       array[baseIndex] = NULL;
      }
    }
   nextIndex = (baseIndex + 1) % size;
   //only change baseIndex if there is a next one
   //if baseIndex is the only one left, do not change
   cerr<<"base in"<<baseIndex<<" rear"<<rearIndex<<endl;
   if(baseIndex != rearIndex)
    baseIndex = nextIndex;
   else
    {
     baseIndex =-1;
     rearIndex =-1;
    }
  }

  /**
   * @brief user should not delete the pointer, use deleteEnd 
   */
  Type& operator[] (uint32 offset)
   {
   uint32 realIndex= (baseIndex + offset) % size;
   return array[realIndex];
   }
  
  /**
   * @brief Return the oldest element in the queue
   * @return The oldest element in the queue 
   */
  Type& getOldest()
  {
   return array[baseIndex];
  }
  
  int getBaseIndex()
  {
   return baseIndex;
  }
  int getRearIndex()
  {
   return rearIndex;
  }
  uint32 getSize()
  {
   return size;
  }
  /**
   * @breif Return the newest element in the queue
   * 
   */
  
  Type& getNewest()
  {
   return array[rearIndex];
   
  }
  
  /**
   * @brief Get the length of the queue, which is the number of elements
   * inside the queue
   */
  uint32 getLen()
  {
   if(rearIndex ==-1 && baseIndex==-1)
    return 0;
   if(rearIndex >= baseIndex)
    return (rearIndex - baseIndex +1);
   else
    return (size+rearIndex-baseIndex +1); 
  }
  /**
   * @brief Clean the queue, will check from the oldest to the nearest. 
   * If pointers stored, and not NULL, will call delete to delete the pointers, 
   * and set it NULL.
   */
  void cleanQueue()
  {
   int i;
   
   for (i= baseIndex; i!=rearIndex; i= ((i+1) %size))
    {
     cerr<<"cleaning" <<i<<endl;
     if(array[i] != NULL)
      {
       delete array[i];
       array[i] = NULL;
      }
    }
   
   if(baseIndex !=-1 && rearIndex !=-1 && array[i] != NULL)
    {
     cerr<<"clearn rear"<<rearIndex<<endl;
     
     cerr<<"still inside, should be the rearIndex" <<i<<endl;
     
     delete array[i];
     array[i] = NULL;
    }
   
  }
  
  
  //    CircularQueue(const CircularQueue& orig) {};
  virtual ~CircularQueue() {
   //make sure RTPFactory first call cleanQueue;
   if(pointersStored)
    {
     for(int i=0; i<size; i++)
      {
       if(array[i] != NULL)
        {
         cerr<<"@@@@@@@should not see this : "<<i<<"not cleaned"<<endl;
         delete array[i];
         array[i] = NULL;
        }
       
      }
    }
   free(array);
  }
  private:
   Type* array;
   bool pointersStored; 
   
   int baseIndex; ///< the index to the oldest element
   int rearIndex; ///< the index to the newest element
   
   uint32 size; ///< capacity of the queue
   
   
};

#endif	/* CIRCULARQUEUE_H */

