using ImageMagick;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageConverter
{
  class Program
  {
    static void Main(string[] args)
    {
      GetPixels();
    }

    private static void GetPixels()
    {
      using (MagickImage image = new MagickImage(@"..\..\..\images\input.png"))
      {
        WriteImageHeader(image);
        WritePixels(image, @"..\..\..\images\input.pixels");

        image.WaveletDenoise(new Percentage(50));

        WritePixels(image, @"..\..\..\images\output.pixels");

        image.Write(@"..\..\..\images\output.png");
      }
    }

    private static void WritePixels(MagickImage image, string fileName)
    {
      using (PixelCollection pixels = image.GetPixels())
      {
        ushort[] data = pixels.ToArray();
        byte[] bytes = new byte[data.Length * 2];

        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(fileName, bytes);
      }
    }

    private static void WriteImageHeader(MagickImage image)
    {
      using (FileStream fs = File.OpenWrite(@"..\..\..\image.h"))
      {
        using (StreamWriter writer = new StreamWriter(fs))
        {
          writer.WriteLine($"cl_uint WIDTH={image.Width};");
          writer.WriteLine($"cl_uint HEIGHT={image.Height};");
          writer.WriteLine($"cl_uint CHANNELCOUNT={image.ChannelCount};");
        }
      }
    }
  }
}
