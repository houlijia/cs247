#include "fast_heap.h"

int main(int argc, const char *argv[]) {
  HeapElement<HeapElementSpec> &elm = fast_heap->get(100);
  elm.discard();
}
 
