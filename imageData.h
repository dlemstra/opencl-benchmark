#include "clew.h"

typedef struct
{
  cl_uint
    event_count;

  cl_event
    *events;

  cl_mem
    input,
    output;

  size_t
    size;

  void
    *inputPixels,
    *outputPixels;
} ImageData;