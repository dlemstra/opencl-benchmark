#include <Windows.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "clew.h"
#include "kernel.h"
#include "main.h"

double getTime()
{
  return (double)clock() / CLOCKS_PER_SEC;
}

cl_uint createBuffers(cl_context context, ImageData *image)
{
  cl_uint
    status;

  image->input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, image->size, image->inputPixels, &status);
  if (status == CL_SUCCESS)
    image->output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, image->size, image->outputPixels, &status);

  return status;
}

void releaseEvents(ImageData *image)
{
  size_t
    i;

  for (i = 0; i < image->event_count; i++)
    clReleaseEvent(image->events[i]);

  free(image->events);
  image->events = (cl_event *)NULL;
  image->event_count = 0;
}

void releaseBuffers(cl_command_queue queue, ImageData *image)
{
  void
    *pixels;

  if (image->input != (cl_mem)NULL)
  {
    pixels = clEnqueueMapBuffer(queue, image->input, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, image->size, image->event_count, image->events, (cl_event *)NULL, (cl_int *)NULL);
    assert(pixels == image->inputPixels);

    clReleaseMemObject(image->input);
    image->input = (cl_mem)NULL;
  }

  if (image->output != (cl_mem)NULL)
  {
    pixels = clEnqueueMapBuffer(queue, image->output, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, image->size, image->event_count, image->events, (cl_event *)NULL, (cl_int *)NULL);
    assert(pixels == image->outputPixels);

    clReleaseMemObject(image->output);
    image->output = (cl_mem)NULL;
  }
}

size_t removePeaks(double *durations, size_t count)
{
  double
    value;

  size_t
    i,
    index,
    j,
    peaks;

  if (count < 10)
    return 0;

  peaks = (size_t)floor(count * 0.2);
  printf("\t\tRemoving top and bottom %d peaks.\n", peaks);

  for (i = 0; i < peaks; i++)
  {
    value = 1000000000000;
    for (j = 0; j < count; j++)
    {
      if (durations[j] == -1)
        continue;

      if (durations[j] < value)
      {
        index = j;
        value = durations[j];
      }
    }

    durations[index] = -1;
  }

  for (i = 0; i < peaks; i++)
  {
    value = 0;
    for (j = 0; j < count; j++)
    {
      if (durations[j] == -1)
        continue;

      if (durations[j] > value)
      {
        index = j;
        value = durations[j];
      }
    }

    durations[index] = -1;
  }

  return peaks;
}

cl_uint runBenchmark(cl_context context, cl_command_queue queue, cl_kernel kernel, ImageData *image)
{
  cl_uint
    status;

  double
    end,
    start,
    *durations,
    total;

  size_t
    i,
    count,
    peaks;

  count = 100;

  durations = (double *)malloc(count * sizeof(*durations));
  for (i = 0; i < count; i++)
  {
    status = createBuffers(context, image);
    if (status == CL_SUCCESS)
    {
      status = setKernelArguments(kernel, image);
      if (status == CL_SUCCESS)
      {
        if (i > 0)
          printf("\r");

        printf("\t\tRunning benchmarks %d%%", (int)(((i + 1) / (double)count) * 100));
        if (i == count - 1)
          printf("\n");

        start = getTime();

        status = runKernel(queue, kernel, image);

        releaseBuffers(queue, image);

        end = getTime();

        durations[i] = end - start;
      }
      else
        releaseBuffers(queue, image);
    }

    releaseEvents(image);

    if (status != CL_SUCCESS)
    {
      printf("\n");
      free(durations);
      return status;
    }
  }

  peaks = removePeaks(durations, count);

  total = 0;
  for (i = 0; i < count; i++)
  {
    if (durations[i] != -1)
      total += durations[i];
  }

  printf("\t\tAverage: %.20g \n", total / (count - (2 * peaks)));

  free(durations);
  return status;
}

cl_int createProgram(cl_context context, cl_device_id device, cl_program *program)
{
  cl_int
    status;

  size_t
    length;

  length = strlen(kernelData);
  *program = clCreateProgramWithSource(context, 1, &kernelData, &length, &status);
  if (status == CL_SUCCESS)
  {
    clBuildProgram(*program, 1, &device, kernelOptions, NULL, NULL);
    if (status != CL_SUCCESS)
      clReleaseProgram(*program);
  }

  return status;
}

void printKernelError(cl_program program, cl_device_id device)
{
  char
    *log;

  size_t
    logSize;

  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
  log = (char*)malloc(logSize);
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, &logSize);
  printf(log);
  free(log);
}

cl_int benchmarkKernel(cl_context context, cl_device_id device, cl_command_queue queue, ImageData *image)
{
  cl_int
    status;

  cl_kernel
    kernel;

  cl_program
    program;

  status = createProgram(context, device, &program);
  if (status != CL_SUCCESS)
    return status;

  printf("\t\tCreated program.\n");

  kernel = clCreateKernel(program, kernelName, &status);
  if (status == CL_SUCCESS)
  {
    printf("\t\tCreated kernel.\n");

    status = runBenchmark(context, queue, kernel, image);

    if (status != CL_SUCCESS)
      printf("\t\tBENCHMARK FAILED.\n");

    clReleaseKernel(kernel);
  }
  else
    printKernelError(program, device);

  clReleaseProgram(program);

  return status;
}

cl_int benchmarkDevice(cl_platform_id platform, cl_device_id device, ImageData *image)
{
  cl_command_queue
    queue;

  cl_context
    context;

  cl_context_properties
    properties[3];

  cl_int
    status;

  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties)(platform);
  properties[2] = 0;

  context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
  if (status != CL_SUCCESS)
    return status;

  printf("\t\tCreated context.\n");

  queue = clCreateCommandQueue(context, device, 0, &status);
  if (status == CL_SUCCESS)
  {
    printf("\t\tCreated command queue.\n");

    status = benchmarkKernel(context, device, queue, image);
    clReleaseCommandQueue(queue);
  }

  clReleaseContext(context);

  return status;
}

cl_int benchmarkPlatform(cl_platform_id platform, ImageData *image)
{
  cl_device_id
    *devices;

  cl_int
    status;

  cl_uint
    numDevices;

  size_t
    i;

  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
  if (status != CL_SUCCESS)
    return status;

  devices = (cl_device_id *)malloc(sizeof(*devices) * numDevices);
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
  if (status == CL_SUCCESS)
  {
    for (i = 0; i < numDevices; ++i)
    {
      char
        buff[160];

      status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 160, buff, NULL);
      if (status != CL_SUCCESS)
        continue;

      printf("\tDevice %d: %s\n", i, buff);

      status = benchmarkDevice(platform, devices[i], image);
      if (status != CL_SUCCESS)
        break;
      break;
    }
  }

  free(devices);

  return status;
}

ImageData* readImage(const char *name)
{
  FILE
    *file;

  ImageData
    *image;

  size_t
    extent,
    size;

  fopen_s(&file, name, "rb");
  if (file == (FILE*)NULL)
    return (ImageData*)NULL;

  image = (ImageData *)malloc(sizeof(*image));
  memset(image, 0, sizeof(*image));

  fseek(file, 0, SEEK_END);
  size = ftell(file) * sizeof(char);

  extent = (((size)+((64) - 1)) & ~((64) - 1));

  image->inputPixels = _aligned_malloc(extent, 4096);
  fread(image->inputPixels, size, 1, file);
  fclose(file);

  image->outputPixels = _aligned_malloc(extent, 4096);

  image->size = extent;

  return image;
}

void freeImage(ImageData *image)
{
  _aligned_free(image->inputPixels);
  _aligned_free(image->outputPixels);
  free(image);
}

int wmain(int argc, wchar_t *argv[])
{
  cl_int
    status;

  cl_platform_id
    *platforms;

  cl_uint
    numPlatforms;

  size_t
    i;

  ImageData
    *image;

  if (clewInit("OpenCL.dll") != CL_SUCCESS)
    return;

  printf("OpenCL initialized.\n");

  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS || numPlatforms == 0)
    return;

  printf("Number of platforms: %d.\n", (int)numPlatforms);

  platforms = (cl_platform_id *)malloc(sizeof(*platforms) * numPlatforms);
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);
  if (status != CL_SUCCESS)
  {
    free(platforms);
    return;
  }

  image = readImage("..\\images\\input.pixels");
  if (image == (ImageData*)NULL)
    image = readImage("images\\input.pixels");

  if (image == (ImageData*)NULL)
    return;

  printf("Input data loaded.\n");

  for (i = 0; i < numPlatforms; ++i)
  {
    char
      buff[160],
      buff2[160];

    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 160, buff, NULL);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 160, buff2, NULL);
    printf("Platform %d: %s %s\n", i, buff, buff2);

    if (benchmarkPlatform(platforms[i], image) != CL_SUCCESS)
      break;
  }

  freeImage(image);
  free(platforms);
}