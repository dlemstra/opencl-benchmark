#include "../image.h"
#include "../imageData.h"
#include "../clew.h"
#include "shared.h"

const char *kernelData =

/*
  Define declarations.
*/
#define OPENCL_DEFINE(VAR,...)	"\n #""define " #VAR " " #__VA_ARGS__ " \n"
#define OPENCL_ELIF(...)	"\n #""elif " #__VA_ARGS__ " \n"
#define OPENCL_ELSE()		"\n #""else " " \n"
#define OPENCL_ENDIF()		"\n #""endif " " \n"
#define OPENCL_IF(...)		"\n #""if " #__VA_ARGS__ " \n"
#define STRINGIFY(...) #__VA_ARGS__ "\n"

/*
  Define declarations.
*/
  OPENCL_DEFINE(GetPixelAlpha(pixel),(QuantumRange-(pixel).w))
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
       PolynomialFunction,
       SinusoidFunction,
       ArcsinFunction,
       ArctanFunction
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
    HanningWeightingFunction,
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
       UndefinedChannel,
       RedChannel = 0x0001,
       GrayChannel = 0x0001,
       CyanChannel = 0x0001,
       GreenChannel = 0x0002,
       MagentaChannel = 0x0002,
       BlueChannel = 0x0004,
       YellowChannel = 0x0004,
       AlphaChannel = 0x0008,
       OpacityChannel = 0x0008,
       MatteChannel = 0x0008,     /* deprecated */
       BlackChannel = 0x0020,
       IndexChannel = 0x0020,
       CompositeChannels = 0x002F,
       AllChannels = 0x7ffffff,
       /*
       Special purpose channel types.
       */
       TrueAlphaChannel = 0x0040, /* extract actual alpha channel from opacity */
       RGBChannels = 0x0080,      /* set alpha from  grayscale mask in RGB */
       GrayChannels = 0x0080,
       SyncChannels = 0x0100,     /* channels should be modified equally */
       DefaultChannels = ((AllChannels | SyncChannels) &~ OpacityChannel)
     } ChannelType;
  )

/*
  Helper functions.
*/

OPENCL_IF((MAGICKCORE_QUANTUM_DEPTH == 8))

  STRINGIFY(
    inline CLQuantum ScaleCharToQuantum(const unsigned char value)
    {
      return((CLQuantum) value);
    }
  )

OPENCL_ELIF((MAGICKCORE_QUANTUM_DEPTH == 16))

  STRINGIFY(
    inline CLQuantum ScaleCharToQuantum(const unsigned char value)
    {
      return((CLQuantum) (257.0f*value));
    }
  )

OPENCL_ELIF((MAGICKCORE_QUANTUM_DEPTH == 32))

  STRINGIFY(
    inline CLQuantum ScaleCharToQuantum(const unsigned char value)
    {
      return((CLQuantum) (16843009.0*value));
    }
  )

OPENCL_ELIF((MAGICKCORE_QUANTUM_DEPTH == 64))

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
      inline int ClampToCanvasWithHalo(const int offset, const int range, const int edge, const int section)
  {
    return clamp(offset, section ? (int)(0 - edge) : (int)0, section ? (range - 1) : (range - 1 + edge));
  }
  )

    STRINGIFY(
      inline CLQuantum ClampToQuantum(const float value)
  {
    return (CLQuantum)(clamp(value, 0.0f, (float)QuantumRange) + 0.5f);
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

  inline CLQuantum getOpacity(CLPixelType p) { return p.w; }
  inline void setOpacity(CLPixelType* p, CLQuantum value) { (*p).w = value; }
  inline float getOpacityF4(float4 p) { return p.w; }
  inline void setOpacityF4(float4* p, float value) { (*p).w = value; }

  inline void setGray(CLPixelType* p, CLQuantum value) { (*p).z = value; (*p).y = value; (*p).x = value; }

  inline float GetPixelIntensity(const int method, const int colorspace, CLPixelType p)
  {
    float red = getRed(p);
    float green = getGreen(p);
    float blue = getBlue(p);

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
  )









    STRINGIFY(
      inline int mirrorBottom(int value)
      {
          return (value < 0) ? - (value) : value;
      }
      inline int mirrorTop(int value, int width)
      {
          return (value >= width) ? (2 * width - value - 1) : value;
      })


  STRINGIFY(
    __kernel __attribute__((reqd_work_group_size(64, 4, 1)))
    void WaveletDenoise(__global CLPixelType *srcImage, __global CLPixelType *dstImage,
					const float threshold,
					const int passes,
					const int imageWidth,
					const int imageHeight)
    {
        const uint workYsize = 4;
        const uint maxPasses = 5;
        const int pad = (1 << (maxPasses - 1));
        const int tileSize = 64;
        const int tileRowPixels = tileSize;

        const float noise[] = { 0.8002, 0.2735, 0.1202, 0.0585, 0.0291, 0.0152, 0.0080, 0.0044 };

        CLQuantum stage[4][16];
        local float buffer[64 * 64];
        int srcx = (get_group_id(0) + get_global_offset(0) / tileSize) * (tileSize - 2 * pad) - pad + get_local_id(0);
        int srcy = (get_group_id(1) + get_global_offset(1) / 4) * (tileSize - 2 * pad) - pad;

        for (uint i = get_local_id(1), j = 0; i < tileSize; i += workYsize, j++)
        {
            CLPixelType pix = srcImage[mirrorTop(mirrorBottom(srcx), imageWidth) + (mirrorTop(mirrorBottom(srcy + i), imageHeight)) * imageWidth];
            stage[0][j] = pix.s0;
            stage[1][j] = pix.s1;
            stage[2][j] = pix.s2;
            stage[3][j] = pix.s3;
        }

        for (int channel = 0; channel < 3; ++channel)
        {
            for (uint i = get_local_id(1), j = 0; i < tileSize; i += workYsize, j++)
            {
                buffer[get_local_id(0) + i * tileRowPixels] = convert_float(stage[channel][j]);
            }

            float tmp[16];
            float accum[16] = { 0 };
            float pixel;
            for (int pass = 0; pass < maxPasses; ++pass)
            {
                const int radius = 1 << pass;
                const int x = get_local_id(0);
                const float thresh = threshold * noise[pass];

                for (uint i = get_local_id(1), j = 0; i < tileSize; i += workYsize, j++)
                {
                    const int offset = i * tileRowPixels;
                    if (pass == 0)
                        tmp[j] = buffer[x + offset];

                    pixel = 0.5f * tmp[j] + 0.25 * (buffer[mirrorBottom(x - radius) + offset] + buffer[mirrorTop(x + radius, tileSize) + offset]);
                    barrier(CLK_LOCAL_MEM_FENCE);
                    buffer[x + offset] = pixel;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                for (uint i = get_local_id(1), j = 0; i < tileSize; i += workYsize, j++)
                {
                    pixel = 0.5f * buffer[x + i * tileRowPixels] + 0.25 * (buffer[x + mirrorBottom(i - radius) * tileRowPixels] + buffer[x + mirrorTop(i + radius, tileRowPixels) * tileRowPixels]);
                    float delta = tmp[j] - pixel;
                    tmp[j] = pixel;
                    if (delta < -thresh) delta += thresh;
                    else if (delta > thresh) delta -= thresh;
                    else delta = 0;
                    accum[j] += delta;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (pass < maxPasses - 1) {
                    for (uint i = get_local_id(1), j = 0; i < tileSize; i += workYsize, j++)
                        buffer[x + i * tileRowPixels] = tmp[j];
                }
                else {
                    for (uint i = get_local_id(1), j = 0; i < tileSize; i += workYsize, j++)
                        accum[j] += tmp[j];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            for (uint j = 0; j < 16; j++)
            {
                stage[channel][j] = ClampToQuantum(accum[j]);
            }

        }

        if ((get_local_id(0) >= pad) && (get_local_id(0) < tileSize - pad) && (srcx >= 0) && (srcx < imageWidth))
        {
            for (uint i = get_local_id(1), j = 0; i < tileSize; i += workYsize, j++)
            {
                if ((i >= pad) && (i < tileSize - pad) && (srcy + i < imageHeight))
                {
                    CLPixelType pix = (CLPixelType)(stage[0][j], stage[1][j], stage[2][j], stage[3][j]);
                    dstImage[srcx + (srcy + i) * imageWidth] = pix;
                }
            }
        }
    }
)
;

#define kernelName "WaveletDenoise"

cl_uint setKernelArguments(cl_kernel kernel, ImageData *image)
{
  cl_uint
    status;

  size_t
    i;

  const cl_int
    PASSES = 5;

  cl_float
    thresh;

  if (CHANNELCOUNT != 4)
    return CL_IMAGE_FORMAT_MISMATCH;

  thresh = 65535 / 2.0;

  i = 0;
  status = clSetKernelArg(kernel, i++, sizeof(cl_mem), (void *)&image->input);
  status |= clSetKernelArg(kernel, i++, sizeof(cl_mem), (void *)&image->output);
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