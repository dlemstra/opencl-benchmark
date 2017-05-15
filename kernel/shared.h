static void RegisterEvent(ImageData *image, cl_event event)
{
  if (image->events == (cl_event *)NULL)
  {
    image->events = (cl_event *)malloc(sizeof(*image->events));
    image->event_count = 1;
  }
  else
    image->events = realloc(image->events, ++image->event_count * sizeof(*image->events));
  if (image->events == (cl_event *)NULL)
    return;
  image->events[image->event_count - 1] = event;
  clRetainEvent(event);
}

#define kernelOptions "-cl-single-precision-constant -cl-mad-enable " \
  "-DCLQuantum=ushort -DCLSignedQuantum=short -DCLPixelType=ushort4 -DQuantumRange=65535.0f "\
  "-DQuantumScale=0.00001525902 -DCharQuantumScale=257.0f -DMagickEpsilon=1.0e-15 -DMagickPI=3.14159265358979323846264338327950288419716939937510 "\
  "-DMaxMap=65535UL -DMAGICKCORE_QUANTUM_DEPTH=16"

#define OPENCL_DEFINE(VAR,...)	"\n #""define " #VAR " " #__VA_ARGS__ " \n"
#define OPENCL_ELIF(...)	"\n #""elif " #__VA_ARGS__ " \n"
#define OPENCL_ELSE()		"\n #""else " " \n"
#define OPENCL_ENDIF()		"\n #""endif " " \n"
#define OPENCL_IF(...)		"\n #""if " #__VA_ARGS__ " \n"
#define STRINGIFY(...) #__VA_ARGS__ "\n"