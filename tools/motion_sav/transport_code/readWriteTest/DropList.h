/* 
 * File:   DropList.h
 * Author: Jianwei Liu 
 *
 * Created on July 18, 2014, 9:00 AM
 */

#ifndef DROPLIST_H
#define	DROPLIST_H

//#ifdef __cplusplus
//extern "C"
//{
//#endif
 
#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>

using std::vector;
using std::cout;
using std::endl;

class AbstractDropList
{
public:
 virtual bool isInside(int checkNumber) = 0;
};

class UniformDropList :public AbstractDropList
{
public:
 UniformDropList(int mod0) :mod(mod0) {};

 bool isInside(int checkNumber) {
  if(checkNumber % mod ==0)
   return true;
  else
   return false;
 }
private:
 int mod;
};

class DropList : public AbstractDropList
{
public:
 DropList (FILE* fp)
 {
  int drop;
  if(fp != NULL)
   {
    while(fscanf(fp, "%d", &drop) != EOF)
     {
        dropArray.push_back(drop);
     }
   }
  else
   {
    cout<<"empty drop list"<<endl;
   }
 }

 bool isInside(int checkNumber)
 {
std::vector<int>::iterator it;
  it = find (dropArray.begin(), dropArray.end(), checkNumber);
  if(it ==  dropArray.end())
   return false;
  else
   return true;
}
 
 //DropList (const DropList& orig);
 virtual ~DropList () {};
 vector<int> dropArray;
private:

};


//#ifdef __cplusplus
//}
//#endif


#endif	/* DROPLIST_H */

