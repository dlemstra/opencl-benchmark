#include "image.h"
#include "imageData.h"
#include "clew.h"

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

const char *kernelData =

/*
  Define declarations.
*/
OPENCL_DEFINE(SigmaUniform, (attenuate*0.015625f))
OPENCL_DEFINE(SigmaGaussian, (attenuate*0.015625f))
OPENCL_DEFINE(SigmaImpulse, (attenuate*0.1f))
OPENCL_DEFINE(SigmaLaplacian, (attenuate*0.0390625f))
OPENCL_DEFINE(SigmaMultiplicativeGaussian, (attenuate*0.5f))
OPENCL_DEFINE(SigmaPoisson, (attenuate*12.5f))
OPENCL_DEFINE(SigmaRandom, (attenuate))
OPENCL_DEFINE(TauGaussian, (attenuate*0.078125f))
OPENCL_DEFINE(MagickMax(x, y), (((x) > (y)) ? (x) : (y)))
OPENCL_DEFINE(MagickMin(x, y), (((x) < (y)) ? (x) : (y)))

  /*
    Typedef declarations.
  */
  STRINGIFY(
    typedef enum
{
  UndefinedColorspace,
  RGBColorspace,            /* Linear RGB colorspace */
  GRAYColorspace,           /* greyscale (linear) image (faked 1 channel) */
  TransparentColorspace,
  OHTAColorspace,
  LabColorspace,
  XYZColorspace,
  YCbCrColorspace,
  YCCColorspace,
  YIQColorspace,
  YPbPrColorspace,
  YUVColorspace,
  CMYKColorspace,           /* negared linear RGB with black separated */
  sRGBColorspace,           /* Default: non-lienar sRGB colorspace */
  HSBColorspace,
  HSLColorspace,
  HWBColorspace,
  Rec601LumaColorspace,
  Rec601YCbCrColorspace,
  Rec709LumaColorspace,
  Rec709YCbCrColorspace,
  LogColorspace,
  CMYColorspace,            /* negated linear RGB colorspace */
  LuvColorspace,
  HCLColorspace,
  LCHColorspace,            /* alias for LCHuv */
  LMSColorspace,
  LCHabColorspace,          /* Cylindrical (Polar) Lab */
  LCHuvColorspace,          /* Cylindrical (Polar) Luv */
  scRGBColorspace,
  HSIColorspace,
  HSVColorspace,            /* alias for HSB */
  HCLpColorspace,
  YDbDrColorspace
} ColorspaceType;
)

STRINGIFY(
  typedef enum
{
  UndefinedCompositeOp,
  NoCompositeOp,
  ModulusAddCompositeOp,
  AtopCompositeOp,
  BlendCompositeOp,
  BumpmapCompositeOp,
  ChangeMaskCompositeOp,
  ClearCompositeOp,
  ColorBurnCompositeOp,
  ColorDodgeCompositeOp,
  ColorizeCompositeOp,
  CopyBlackCompositeOp,
  CopyBlueCompositeOp,
  CopyCompositeOp,
  CopyCyanCompositeOp,
  CopyGreenCompositeOp,
  CopyMagentaCompositeOp,
  CopyOpacityCompositeOp,
  CopyRedCompositeOp,
  CopyYellowCompositeOp,
  DarkenCompositeOp,
  DstAtopCompositeOp,
  DstCompositeOp,
  DstInCompositeOp,
  DstOutCompositeOp,
  DstOverCompositeOp,
  DifferenceCompositeOp,
  DisplaceCompositeOp,
  DissolveCompositeOp,
  ExclusionCompositeOp,
  HardLightCompositeOp,
  HueCompositeOp,
  InCompositeOp,
  LightenCompositeOp,
  LinearLightCompositeOp,
  LuminizeCompositeOp,
  MinusDstCompositeOp,
  ModulateCompositeOp,
  MultiplyCompositeOp,
  OutCompositeOp,
  OverCompositeOp,
  OverlayCompositeOp,
  PlusCompositeOp,
  ReplaceCompositeOp,
  SaturateCompositeOp,
  ScreenCompositeOp,
  SoftLightCompositeOp,
  SrcAtopCompositeOp,
  SrcCompositeOp,
  SrcInCompositeOp,
  SrcOutCompositeOp,
  SrcOverCompositeOp,
  ModulusSubtractCompositeOp,
  ThresholdCompositeOp,
  XorCompositeOp,
  /* These are new operators, added after the above was last sorted.
   * The list should be re-sorted only when a new library version is
   * created.
   */
  DivideDstCompositeOp,
  DistortCompositeOp,
  BlurCompositeOp,
  PegtopLightCompositeOp,
  VividLightCompositeOp,
  PinLightCompositeOp,
  LinearDodgeCompositeOp,
  LinearBurnCompositeOp,
  MathematicsCompositeOp,
  DivideSrcCompositeOp,
  MinusSrcCompositeOp,
  DarkenIntensityCompositeOp,
  LightenIntensityCompositeOp
} CompositeOperator;
)

STRINGIFY(
  typedef enum
{
  UndefinedFunction,
  ArcsinFunction,
  ArctanFunction,
  PolynomialFunction,
  SinusoidFunction
} MagickFunction;
)

STRINGIFY(
  typedef enum
{
  UndefinedNoise,
  UniformNoise,
  GaussianNoise,
  MultiplicativeGaussianNoise,
  ImpulseNoise,
  LaplacianNoise,
  PoissonNoise,
  RandomNoise
} NoiseType;
)

STRINGIFY(
  typedef enum
{
  UndefinedPixelIntensityMethod = 0,
  AveragePixelIntensityMethod,
  BrightnessPixelIntensityMethod,
  LightnessPixelIntensityMethod,
  Rec601LumaPixelIntensityMethod,
  Rec601LuminancePixelIntensityMethod,
  Rec709LumaPixelIntensityMethod,
  Rec709LuminancePixelIntensityMethod,
  RMSPixelIntensityMethod,
  MSPixelIntensityMethod
} PixelIntensityMethod;
)

STRINGIFY(
  typedef enum {
  BoxWeightingFunction = 0,
  TriangleWeightingFunction,
  CubicBCWeightingFunction,
  HannWeightingFunction,
  HammingWeightingFunction,
  BlackmanWeightingFunction,
  GaussianWeightingFunction,
  QuadraticWeightingFunction,
  JincWeightingFunction,
  SincWeightingFunction,
  SincFastWeightingFunction,
  KaiserWeightingFunction,
  WelshWeightingFunction,
  BohmanWeightingFunction,
  LagrangeWeightingFunction,
  CosineWeightingFunction,
} ResizeWeightingFunctionType;
)

STRINGIFY(
  typedef enum
{
  UndefinedChannel = 0x0000,
  RedChannel = 0x0001,
  GrayChannel = 0x0001,
  CyanChannel = 0x0001,
  GreenChannel = 0x0002,
  MagentaChannel = 0x0002,
  BlueChannel = 0x0004,
  YellowChannel = 0x0004,
  BlackChannel = 0x0008,
  AlphaChannel = 0x0010,
  OpacityChannel = 0x0010,
  IndexChannel = 0x0020,             /* Color Index Table? */
  ReadMaskChannel = 0x0040,          /* Pixel is Not Readable? */
  WriteMaskChannel = 0x0080,         /* Pixel is Write Protected? */
  MetaChannel = 0x0100,              /* ???? */
  CompositeChannels = 0x001F,
  AllChannels = 0x7ffffff,
  /*
    Special purpose channel types.
    FUTURE: are these needed any more - they are more like hacks
    SyncChannels for example is NOT a real channel but a 'flag'
    It really says -- "User has not defined channels"
    Though it does have extra meaning in the "-auto-level" operator
  */
  TrueAlphaChannel = 0x0100, /* extract actual alpha channel from opacity */
  RGBChannels = 0x0200,      /* set alpha from grayscale mask in RGB */
  GrayChannels = 0x0400,
  SyncChannels = 0x20000,    /* channels modified as a single unit */
  DefaultChannels = AllChannels
} ChannelType;  /* must correspond to PixelChannel */
)

/*
  Helper functions.
*/

OPENCL_IF((MAGICKCORE_QUANTUM_DEPTH == 8))

STRINGIFY(
  inline CLQuantum ScaleCharToQuantum(const unsigned char value)
{
  return((CLQuantum)value);
}
)

OPENCL_ELIF((MAGICKCORE_QUANTUM_DEPTH == 16))

STRINGIFY(
  inline CLQuantum ScaleCharToQuantum(const unsigned char value)
{
  return((CLQuantum)(257.0f*value));
}
)

OPENCL_ELIF((MAGICKCORE_QUANTUM_DEPTH == 32))

STRINGIFY(
  inline CLQuantum ScaleCharToQuantum(const unsigned char value)
{
  return((CLQuantum)(16843009.0*value));
}
)

OPENCL_ENDIF()

STRINGIFY(
  inline int ClampToCanvas(const int offset, const int range)
{
  return clamp(offset, (int)0, range - 1);
}
)

STRINGIFY(
  inline CLQuantum ClampToQuantum(const float value)
{
  return (CLQuantum)(clamp(value, 0.0f, QuantumRange) + 0.5f);
}
)

STRINGIFY(
  inline uint ScaleQuantumToMap(CLQuantum value)
{
  if (value >= (CLQuantum)MaxMap)
    return ((uint)MaxMap);
  else
    return ((uint)value);
}
)

STRINGIFY(
  inline float PerceptibleReciprocal(const float x)
{
  float sign = x < (float) 0.0 ? (float)-1.0 : (float) 1.0;
  return((sign*x) >= MagickEpsilon ? (float) 1.0 / x : sign*((float) 1.0 / MagickEpsilon));
}
)

STRINGIFY(
  inline float RoundToUnity(const float value)
{
  return clamp(value, 0.0f, 1.0f);
}
)

STRINGIFY(

  inline unsigned int getPixelIndex(const unsigned int number_channels,
    const unsigned int columns, const unsigned int x, const unsigned int y)
{
  return (x * number_channels) + (y * columns * number_channels);
}

inline float getPixelRed(const __global CLQuantum *p) { return (float)*p; }
inline float getPixelGreen(const __global CLQuantum *p) { return (float)*(p + 1); }
inline float getPixelBlue(const __global CLQuantum *p) { return (float)*(p + 2); }
inline float getPixelAlpha(const __global CLQuantum *p, const unsigned int number_channels) { return (float)*(p + number_channels - 1); }

inline void setPixelRed(__global CLQuantum *p, const CLQuantum value) { *p = value; }
inline void setPixelGreen(__global CLQuantum *p, const CLQuantum value) { *(p + 1) = value; }
inline void setPixelBlue(__global CLQuantum *p, const CLQuantum value) { *(p + 2) = value; }
inline void setPixelAlpha(__global CLQuantum *p, const unsigned int number_channels, const CLQuantum value) { *(p + number_channels - 1) = value; }

inline CLQuantum getBlue(CLPixelType p) { return p.x; }
inline void setBlue(CLPixelType* p, CLQuantum value) { (*p).x = value; }
inline float getBlueF4(float4 p) { return p.x; }
inline void setBlueF4(float4* p, float value) { (*p).x = value; }

inline CLQuantum getGreen(CLPixelType p) { return p.y; }
inline void setGreen(CLPixelType* p, CLQuantum value) { (*p).y = value; }
inline float getGreenF4(float4 p) { return p.y; }
inline void setGreenF4(float4* p, float value) { (*p).y = value; }

inline CLQuantum getRed(CLPixelType p) { return p.z; }
inline void setRed(CLPixelType* p, CLQuantum value) { (*p).z = value; }
inline float getRedF4(float4 p) { return p.z; }
inline void setRedF4(float4* p, float value) { (*p).z = value; }

inline CLQuantum getAlpha(CLPixelType p) { return p.w; }
inline void setAlpha(CLPixelType* p, CLQuantum value) { (*p).w = value; }
inline float getAlphaF4(float4 p) { return p.w; }
inline void setAlphaF4(float4* p, float value) { (*p).w = value; }

inline void ReadChannels(const __global CLQuantum *p, const unsigned int number_channels,
  const ChannelType channel, float *red, float *green, float *blue, float *alpha)
{
  if ((channel & RedChannel) != 0)
    *red = getPixelRed(p);

  if (number_channels > 2)
  {
    if ((channel & GreenChannel) != 0)
      *green = getPixelGreen(p);

    if ((channel & BlueChannel) != 0)
      *blue = getPixelBlue(p);
  }

  if (((number_channels == 4) || (number_channels == 2)) &&
    ((channel & AlphaChannel) != 0))
    *alpha = getPixelAlpha(p, number_channels);
}

inline float4 ReadAllChannels(const __global CLQuantum *image, const unsigned int number_channels,
  const unsigned int columns, const unsigned int x, const unsigned int y)
{
  const __global CLQuantum *p = image + getPixelIndex(number_channels, columns, x, y);

  float4 pixel;

  pixel.x = getPixelRed(p);

  if (number_channels > 2)
  {
    pixel.y = getPixelGreen(p);
    pixel.z = getPixelBlue(p);
  }

  if ((number_channels == 4) || (number_channels == 2))
    pixel.w = getPixelAlpha(p, number_channels);
  return(pixel);
}

inline float4 ReadFloat4(const __global CLQuantum *image, const unsigned int number_channels,
  const unsigned int columns, const unsigned int x, const unsigned int y, const ChannelType channel)
{
  const __global CLQuantum *p = image + getPixelIndex(number_channels, columns, x, y);

  float red = 0.0f;
  float green = 0.0f;
  float blue = 0.0f;
  float alpha = 0.0f;

  ReadChannels(p, number_channels, channel, &red, &green, &blue, &alpha);
  return (float4)(red, green, blue, alpha);
}

inline void WriteChannels(__global CLQuantum *p, const unsigned int number_channels,
  const ChannelType channel, float red, float green, float blue, float alpha)
{
  if ((channel & RedChannel) != 0)
    setPixelRed(p, ClampToQuantum(red));

  if (number_channels > 2)
  {
    if ((channel & GreenChannel) != 0)
      setPixelGreen(p, ClampToQuantum(green));

    if ((channel & BlueChannel) != 0)
      setPixelBlue(p, ClampToQuantum(blue));
  }

  if (((number_channels == 4) || (number_channels == 2)) &&
    ((channel & AlphaChannel) != 0))
    setPixelAlpha(p, number_channels, ClampToQuantum(alpha));
}

inline void WriteAllChannels(__global CLQuantum *image, const unsigned int number_channels,
  const unsigned int columns, const unsigned int x, const unsigned int y, float4 pixel)
{
  __global CLQuantum *p = image + getPixelIndex(number_channels, columns, x, y);

  setPixelRed(p, ClampToQuantum(pixel.x));

  if (number_channels > 2)
  {
    setPixelGreen(p, ClampToQuantum(pixel.y));
    setPixelBlue(p, ClampToQuantum(pixel.z));
  }

  if ((number_channels == 4) || (number_channels == 2))
    setPixelAlpha(p, number_channels, ClampToQuantum(pixel.w));
}

inline void WriteFloat4(__global CLQuantum *image, const unsigned int number_channels,
  const unsigned int columns, const unsigned int x, const unsigned int y, const ChannelType channel,
  float4 pixel)
{
  __global CLQuantum *p = image + getPixelIndex(number_channels, columns, x, y);
  WriteChannels(p, number_channels, channel, pixel.x, pixel.y, pixel.z, pixel.w);
}

inline float GetPixelIntensity(const unsigned int colorspace,
  const unsigned int method, float red, float green, float blue)
{
  float intensity;

  if (colorspace == GRAYColorspace)
    return red;

  switch (method)
  {
  case AveragePixelIntensityMethod:
  {
    intensity = (red + green + blue) / 3.0;
    break;
  }
  case BrightnessPixelIntensityMethod:
  {
    intensity = MagickMax(MagickMax(red, green), blue);
    break;
  }
  case LightnessPixelIntensityMethod:
  {
    intensity = (MagickMin(MagickMin(red, green), blue) +
      MagickMax(MagickMax(red, green), blue)) / 2.0;
    break;
  }
  case MSPixelIntensityMethod:
  {
    intensity = (float)(((float)red*red + green*green + blue*blue) /
      (3.0*QuantumRange));
    break;
  }
  case Rec601LumaPixelIntensityMethod:
  {
    /*
    if (image->colorspace == RGBColorspace)
    {
      red=EncodePixelGamma(red);
      green=EncodePixelGamma(green);
      blue=EncodePixelGamma(blue);
    }
    */
    intensity = 0.298839*red + 0.586811*green + 0.114350*blue;
    break;
  }
  case Rec601LuminancePixelIntensityMethod:
  {
    /*
    if (image->colorspace == sRGBColorspace)
    {
      red=DecodePixelGamma(red);
      green=DecodePixelGamma(green);
      blue=DecodePixelGamma(blue);
    }
    */
    intensity = 0.298839*red + 0.586811*green + 0.114350*blue;
    break;
  }
  case Rec709LumaPixelIntensityMethod:
  default:
  {
    /*
    if (image->colorspace == RGBColorspace)
    {
      red=EncodePixelGamma(red);
      green=EncodePixelGamma(green);
      blue=EncodePixelGamma(blue);
    }
    */
    intensity = 0.212656*red + 0.715158*green + 0.072186*blue;
    break;
  }
  case Rec709LuminancePixelIntensityMethod:
  {
    /*
    if (image->colorspace == sRGBColorspace)
    {
      red=DecodePixelGamma(red);
      green=DecodePixelGamma(green);
      blue=DecodePixelGamma(blue);
    }
    */
    intensity = 0.212656*red + 0.715158*green + 0.072186*blue;
    break;
  }
  case RMSPixelIntensityMethod:
  {
    intensity = (float)(sqrt((float)red*red + green*green + blue*blue) /
      sqrt(3.0));
    break;
  }
  }

  return intensity;
}

inline int mirrorBottom(int value)
{
  return (value < 0) ? -(value) : value;
}

inline int mirrorTop(int value, int width)
{
  return (value >= width) ? (2 * width - value - 1) : value;
}
)

STRINGIFY(
    __kernel __attribute__((reqd_work_group_size(64, 4, 1)))
    void WaveletDenoise(__global CLQuantum *srcImage,__global CLQuantum *dstImage,
      const unsigned int number_channels,const unsigned int max_channels,
      const float threshold,const int passes,const unsigned int imageWidth,
      const unsigned int imageHeight)
  {
    const int pad = (1 << (passes - 1));
    const int tileSize = 64;
    const int tileRowPixels = 64;
    const float noise[] = { 0.8002, 0.2735, 0.1202, 0.0585, 0.0291, 0.0152, 0.0080, 0.0044 };

    CLQuantum stage[48]; // 16 * 3 (we only need 3 channels)

    local float buffer[64 * 64];

    int srcx = (get_group_id(0) + get_global_offset(0) / tileSize) * (tileSize - 2 * pad) - pad + get_local_id(0);
    int srcy = (get_group_id(1) + get_global_offset(1) / 4) * (tileSize - 2 * pad) - pad;

    for (int i = get_local_id(1); i < tileSize; i += get_local_size(1)) {
      int pos = (mirrorTop(mirrorBottom(srcx), imageWidth) * number_channels) +
                (mirrorTop(mirrorBottom(srcy + i), imageHeight)) * imageWidth * number_channels;

      for (int channel = 0; channel < max_channels; ++channel)
        stage[(i / 4) + (16 * channel)] = srcImage[pos + channel];
    }

    for (int channel = 0; channel < max_channels; ++channel) {
      // Load LDS
      for (int i = get_local_id(1); i < tileSize; i += get_local_size(1))
        buffer[get_local_id(0) + i * tileRowPixels] = convert_float(stage[(i / 4) + (16 * channel)]);

      // Process

      float tmp[16];
      float accum[16];
      float pixel;

      for (int i = 0; i < 16; i++)
        accum[i]=0.0f;

      for (int pass = 0; pass < passes; ++pass) {
        const int radius = 1 << pass;
        const int x = get_local_id(0);
        const float thresh = threshold * noise[pass];

        // Apply horizontal hat
        for (int i = get_local_id(1); i < tileSize; i += get_local_size(1)) {
          const int offset = i * tileRowPixels;
          if (pass == 0)
            tmp[i / 4] = buffer[x + offset]; // snapshot input on first pass
          pixel = 0.5f * tmp[i / 4] + 0.25 * (buffer[mirrorBottom(x - radius) + offset] + buffer[mirrorTop(x + radius, tileSize) + offset]);
          barrier(CLK_LOCAL_MEM_FENCE);
          buffer[x + offset] = pixel;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Apply vertical hat
        for (int i = get_local_id(1); i < tileSize; i += get_local_size(1)) {
          pixel = 0.5f * buffer[x + i * tileRowPixels] + 0.25 * (buffer[x + mirrorBottom(i - radius) * tileRowPixels] + buffer[x + mirrorTop(i + radius, tileRowPixels) * tileRowPixels]);
          float delta = tmp[i / 4] - pixel;
          tmp[i / 4] = pixel; // hold output in tmp until all workitems are done
          if (delta < -thresh)
            delta += thresh;
          else if (delta > thresh)
            delta -= thresh;
          else
            delta = 0;
          accum[i / 4] += delta;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (pass < passes - 1)
          for (int i = get_local_id(1); i < tileSize; i += get_local_size(1))
            buffer[x + i * tileRowPixels] = tmp[i / 4]; // store lowpass for next pass
        else  // last pass
          for (int i = get_local_id(1); i < tileSize; i += get_local_size(1))
            accum[i / 4] += tmp[i / 4]; // add the lowpass signal back to output
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      for (int i = get_local_id(1); i < tileSize; i += get_local_size(1))
        stage[(i / 4) + (16 * channel)] = ClampToQuantum(accum[i / 4]);

      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write from stage to output

    if ((get_local_id(0) >= pad) && (get_local_id(0) < tileSize - pad) && (srcx >= 0) && (srcx < imageWidth)) {
      for (int i = get_local_id(1); i < tileSize; i += get_local_size(1)) {
        if ((i >= pad) && (i < tileSize - pad) && (srcy + i >= 0) && (srcy + i < imageHeight)) {
          int pos = (srcx * number_channels) + ((srcy + i) * (imageWidth * number_channels));
          for (int channel = 0; channel < max_channels; ++channel) {
            dstImage[pos + channel] = stage[(i / 4) + (16 * channel)];
          }
        }
      }
    }
  }
  /*
  \n #define WD_TILE_SIZE 64 \n
  \n #define WD_Y_SIZE 4 \n
  __kernel __attribute__((reqd_work_group_size(WD_TILE_SIZE, WD_Y_SIZE, 1)))
  void WaveletDenoise(__global CLQuantum *srcImage, __global CLQuantum *dstImage,
    const unsigned int number_channels, const unsigned int max_channels,
    const float threshold, const int passes, const unsigned int imageWidth,
    const unsigned int imageHeight)
{
  const int pad = (1 << (passes - 1));
  const float noise[] = { 0.8002, 0.2735, 0.1202, 0.0585, 0.0291, 0.0152, 0.0080, 0.0044 };

  local CLQuantum stage[3][16];

  local float buffer[WD_TILE_SIZE * WD_TILE_SIZE];

  int srcx = (get_group_id(0) + get_global_offset(0) / WD_TILE_SIZE) * (WD_TILE_SIZE - 2 * pad) - pad + get_local_id(0);
  int srcy = (get_group_id(1) + get_global_offset(1) / 4) * (WD_TILE_SIZE - 2 * pad) - pad;

\n #pragma unroll (16) \n
  for (int i = get_local_id(1); i < WD_TILE_SIZE; i += WD_Y_SIZE) {
    int pos = (mirrorTop(mirrorBottom(srcx), imageWidth) * number_channels) +
      (mirrorTop(mirrorBottom(srcy + i), imageHeight)) * imageWidth * number_channels;

    for (int channel = 0, j = 0; channel < max_channels; channel++, j++)
      stage[channel][j] = srcImage[pos + channel];
  }

  for (int channel = 0; channel < max_channels; ++channel) {
    // Load LDS
\n #pragma unroll (16) \n
    for (int i = get_local_id(1), j = 0; i < WD_TILE_SIZE; i += WD_Y_SIZE, j++)
      buffer[get_local_id(0) + i * WD_TILE_SIZE] = convert_float(stage[channel][j]);

    // Process

    float tmp[16];
    float accum[16];
    float pixel;

    for (int i = 0; i < 16; i++)
      accum[i] = 0.0f;

    for (int pass = 0; pass < passes; ++pass) {
      const int radius = 1 << pass;
      const int x = get_local_id(0);
      const float thresh = threshold * noise[pass];

      // Apply horizontal hat
      for (int i = get_local_id(1), j=0; i < WD_TILE_SIZE; i += WD_Y_SIZE, j++) {
        const int offset = i * WD_TILE_SIZE;
        if (pass == 0)
          tmp[j] = buffer[x + offset]; // snapshot input on first pass
        pixel = 0.5f * tmp[j] + 0.25 * (buffer[mirrorBottom(x - radius) + offset] + buffer[mirrorTop(x + radius, WD_TILE_SIZE) + offset]);
        barrier(CLK_LOCAL_MEM_FENCE);
        buffer[x + offset] = pixel;
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Apply vertical hat
      for (int i = get_local_id(1), j=0; i < WD_TILE_SIZE; i += WD_Y_SIZE, j++) {
        pixel = 0.5f * buffer[x + i * WD_TILE_SIZE] + 0.25 * (buffer[x + mirrorBottom(i - radius) * WD_TILE_SIZE] + buffer[x + mirrorTop(i + radius, WD_TILE_SIZE) * WD_TILE_SIZE]);
        float delta = tmp[j] - pixel;
        tmp[i / 4] = pixel; // hold output in tmp until all workitems are done
        if (delta < -thresh)
          delta += thresh;
        else if (delta > thresh)
          delta -= thresh;
        else
          delta = 0;
        accum[j] += delta;
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      if (pass < passes - 1)
      {
\n #pragma unroll (16) \n
        for (int i = get_local_id(1), j=0; i < WD_TILE_SIZE; i += WD_Y_SIZE, j++)
          buffer[x + i * WD_TILE_SIZE] = tmp[j]; // store lowpass for next pass
      }
      else  // last pass
      {
\n #pragma unroll (16) \n
        for (int i = get_local_id(1), j=0; i < WD_TILE_SIZE; i += WD_Y_SIZE, j++)
          accum[j] += tmp[j]; // add the lowpass signal back to output
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = get_local_id(1), j=0; i < WD_TILE_SIZE; i += WD_Y_SIZE, j++)
      stage[channel][j] = ClampToQuantum(accum[j]);

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write from stage to output

  if ((get_local_id(0) >= pad) && (get_local_id(0) < WD_TILE_SIZE - pad) && (srcx >= 0) && (srcx < imageWidth)) {
\n #pragma unroll (16) \n
    for (int i = get_local_id(1); i < WD_TILE_SIZE; i += WD_Y_SIZE) {
      if ((i >= pad) && (i < WD_TILE_SIZE - pad) && (srcy + i >= 0) && (srcy + i < imageHeight)) {
        int pos = (srcx * number_channels) + ((srcy + i) * (imageWidth * number_channels));
        for (int channel = 0, j=0; channel < max_channels; ++channel, j++) {
          dstImage[pos + channel] = stage[channel][j];
        }
      }
    }
  }
}*/
);

#define kernelName "WaveletDenoise"

cl_uint setKernelArguments(cl_kernel kernel, ImageData *image)
{
  cl_uint
    status;

  size_t
    i;

  const cl_int
    PASSES = 5;

  cl_uint
    max_channels;

  cl_float
    thresh;

  max_channels = CHANNELCOUNT;
  if ((max_channels == 4) || (max_channels == 2))
    max_channels = max_channels - 1;

  thresh = 65535 / 2.0;

  i = 0;
  status = clSetKernelArg(kernel, i++, sizeof(cl_mem), (void *)&image->input);
  status |= clSetKernelArg(kernel, i++, sizeof(cl_mem), (void *)&image->output);
  status |= clSetKernelArg(kernel, i++, sizeof(cl_uint), (void *)&CHANNELCOUNT);
  status |= clSetKernelArg(kernel, i++, sizeof(cl_uint), (void *)&max_channels);
  status |= clSetKernelArg(kernel, i++, sizeof(cl_float), (void *)&thresh);
  status |= clSetKernelArg(kernel, i++, sizeof(cl_int), (void *)&PASSES);
  status |= clSetKernelArg(kernel, i++, sizeof(cl_uint), (void *)&WIDTH);
  status |= clSetKernelArg(kernel, i++, sizeof(cl_uint), (void *)&HEIGHT);

  return status;
}

cl_uint runKernel(cl_command_queue queue, cl_kernel kernel, ImageData *image)
{
  cl_uint
    status;

  cl_event
    event;

  size_t
    goffset[2],
    gsize[2],
    lsize[2],
    passes,
    x;

  const cl_int
    PASSES = 5;

  const int
    TILESIZE = 64,
    PAD = 1 << (PASSES - 1),
    SIZE = TILESIZE - 2 * PAD;

  passes = (((1.0f*WIDTH)*HEIGHT) + 1999999.0f) / 2000000.0f;
  passes = (passes < 1) ? 1 : passes;

  for (x = 0; x < passes; ++x)
  {
    gsize[0] = ((WIDTH + (SIZE - 1)) / SIZE)*TILESIZE;
    gsize[1] = ((((HEIGHT + (SIZE - 1)) / SIZE) + passes - 1) / passes) * 4;
    lsize[0] = TILESIZE;
    lsize[1] = 4;
    goffset[0] = 0;
    goffset[1] = x*gsize[1];

    status = clEnqueueNDRangeKernel(queue, kernel, 2, goffset, gsize, lsize, image->event_count, image->events, &event);
    if (status != CL_SUCCESS)
      return status;

    RegisterEvent(image, event);
    clReleaseEvent(event);

    clFlush(queue);
  }

  return status;
}