#include <dirent.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <chrono>
#include <float.h>
#include <limits.h>
#include "nvtx.cuh"
#include <math.h>

#include "cuZFP.h"

#include "zfp/bitstream.inl"

using namespace std;
using std::cout;
typedef unsigned long long Word;

// "CUDA version currently only supports 64bit words
typedef uint64 bitstream_word;

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef struct
{
  float *h_uncompressed_data;
  float *d_uncompressed_data;
  float *h_decompressed_data;
  float *d_decompressed_data;
  bitstream_word *h_compressed_data;
  bitstream_word *d_compressed_data;
} Data_t;

void zfp_stream_rewind_device(bitstream *s)
{
  s->ptr = s->begin;
  s->buffer = 0;
  s->bits = 0;
}

#ifdef BIT_STREAM_STRIDED
/* set block size in number of words and spacing in number of blocks */
inline_ int
stream_set_stride_device(bitstream *s, size_t block, ptrdiff_t delta)
{
  /* ensure block size is a power of two */
  if (block & (block - 1))
    return 0;
  s->mask = block - 1;
  s->delta = delta * block;
  return 1;
}
#endif

bitstream *stream_open_device(void *buffer, size_t bytes)
{
  bitstream *s = (bitstream *)malloc(sizeof(bitstream));
  if (s)
  {
    s->begin = (bitstream_word *)buffer;
    s->end = s->begin + bytes / sizeof(bitstream_word);
#ifdef BIT_STREAM_STRIDED
    stream_set_stride_device(s, 0, 0);
#endif
    zfp_stream_rewind_device(s);
  }
  return s;
}

size_t compress(Data_t *data, size_t nx, size_t ny, size_t nz,
                int rate)
{
  zfp_field *field;       /* array meta data */
  zfp_stream *zfp;        /* compressed stream */
  size_t compressed_size; /* byte size of compressed buffer */
  bitstream *stream;      /* bit stream to write to or read from */
  size_t zfpsize;         /* byte size of compressed stream */

  /* allocate meta data for the 3D array a[nz][ny][nx] */
  field = zfp_field_3d(data->d_uncompressed_data, zfp_type_float, nx, ny, nz);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(NULL);

  /* set compression mode and parameters via one of four functions */
  zfp_stream_set_rate(zfp, rate, zfp_type_float, zfp_field_dimensionality(field),
                      zfp_false);

  /* allocate buffer for compressed data */
  compressed_size = zfp_stream_maximum_size(zfp, field);

  cudaMalloc(&data->d_compressed_data, compressed_size);

  stream = stream_open_device(data->d_compressed_data, compressed_size);

  zfp_stream_set_bit_stream(zfp, stream);

  zfp_stream_rewind_device(zfp->stream);

  /* compress array and output compressed stream */
  NVTX_PUSH_RANGE("ZFP_COMPRESS", MY_YELLOW);
  zfpsize = cuda_compress(zfp, field);
  NVTX_POP_RANGE();

  if (!zfpsize)
  {
    fprintf(stderr, "compression failed\n");
  }

  /* clean up */
  zfp_field_free(field);
  zfp_stream_close(zfp);

  stream_close(stream);

  return compressed_size;
}

void decompress(Data_t *data, size_t compressed_size, size_t nx, size_t ny, size_t nz, int rate)
{
  zfp_field *field;  /* array meta data */
  zfp_stream *zfp;   /* compressed stream */
  bitstream *stream; /* bit stream to write to or read from */

  /* allocate meta data for the 3D array a[nz][ny][nx] */
  field = zfp_field_3d(data->d_decompressed_data, zfp_type_float, nx, ny, nz);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(NULL);

  /* set compression mode and parameters via one of four functions */
  zfp_stream_set_rate(zfp, rate, zfp_type_float, zfp_field_dimensionality(field),
                      zfp_false);

  /* allocate buffer for compressed data */
  stream = stream_open_device(data->d_compressed_data, compressed_size);

  zfp_stream_set_bit_stream(zfp, stream);

  zfp_stream_rewind_device(zfp->stream);

  /* compress array and output compressed stream */
  NVTX_PUSH_RANGE("ZFP_DECOMPRESS", MY_YELLOW);
  cuda_decompress(zfp, field);
  NVTX_POP_RANGE();

  /* clean up */
  zfp_field_free(field);
  zfp_stream_close(zfp);
}

int checkIfInputIsCorrupted(float *h_input_before_compression, float *d_input_after_compression, size_t len)
{
  float *h_input_after_compression;
  cudaMallocHost(&h_input_after_compression, len * sizeof(float));
  cudaMemcpy(h_input_after_compression, d_input_after_compression, len * sizeof(float), cudaMemcpyDeviceToHost);

  int errors = 0;
  for (size_t i = 0; i < len; i++)
  {
    if (h_input_before_compression[i] != h_input_after_compression[i])
    {
      // cout << i << " Error here. Before Compression: " << h_input_before_compression[i] << " . After compression: " << h_input_after_compression[i] << "\n";
      errors++;
    }
  }

  if (errors > 0)
  {
    fprintf(stderr, "%i errors happened in the input data after compression\n", errors);
  }
  else
  {
    fprintf(stderr, "No errors have been found in the input data after compression.\n");
  }

  cudaFreeHost(&h_input_after_compression);

  return 0;
}

static void
print_error(const void *fin, const void *fout, zfp_type type, size_t n)
{
  // Code from: https://github.com/LLNL/zfp/blob/fc96c9158e8befc227f9f46093f5e1b35723be02/utils/zfp.c#L30-L81
  const int32 *i32i = (const int32 *)fin;
  const int64 *i64i = (const int64 *)fin;
  const float *f32i = (const float *)fin;
  const double *f64i = (const double *)fin;
  const int32 *i32o = (const int32 *)fout;
  const int64 *i64o = (const int64 *)fout;
  const float *f32o = (const float *)fout;
  const double *f64o = (const double *)fout;
  double fmin = +DBL_MAX;
  double fmax = -DBL_MAX;
  double erms = 0;
  double ermsn = 0;
  double emax = 0;
  double psnr = 0;
  size_t i;

  for (i = 0; i < n; i++)
  {
    double d, val;
    switch (type)
    {
    case zfp_type_int32:
      d = fabs((double)(i32i[i] - i32o[i]));
      val = (double)i32i[i];
      break;
    case zfp_type_int64:
      d = fabs((double)(i64i[i] - i64o[i]));
      val = (double)i64i[i];
      break;
    case zfp_type_float:
      d = fabs((double)(f32i[i] - f32o[i]));
      val = (double)f32i[i];
      break;
    case zfp_type_double:
      d = fabs(f64i[i] - f64o[i]);
      val = f64i[i];
      break;
    default:
      return;
    }
    emax = MAX(emax, d);
    erms += d * d;
    fmin = MIN(fmin, val);
    fmax = MAX(fmax, val);
  }
  erms = sqrt(erms / n);
  ermsn = erms / (fmax - fmin);
  psnr = 20 * log10((fmax - fmin) / (2 * erms));
  fprintf(stderr, "Stats = rmse=%.4g nrmse=%.4g maxe=%.4g psnr=%.2f\n\n", erms, ermsn, emax, psnr);
}

void readInputDataFromFile(string filepath, float *h_array, size_t len)
{
  std::ifstream ifs(filepath.c_str(), std::ios::binary | std::ios::in);
  if (not ifs.is_open())
  {
    std::cerr << "fail to open " << filepath << std::endl;
    exit(1);
  }
  ifs.read(reinterpret_cast<char *>(h_array), std::streamsize(len * sizeof(float)));
  ifs.close();
}

void exportData(string path, void *h_data, int data_size, size_t len)
{
  auto file = fopen(path.c_str(), "wb");
  fwrite(h_data, data_size, len, file);
  fclose(file);
}

int main(int argc, char *argv[])
{
  int gpus = 4;
  int iterationsPerGpu = 10;
  string inputFilepath = "../../hurr-CLOUDf48-500x500x100";
  size_t nx = 500;
  size_t ny = 500;
  size_t nz = 100;
  bool dumpData = false;
  int rate = 8;

  if (argc < 8 && argc > 1)
  {
    cout << "Wrong number of parameters. Expected parameters:\n";
    cout << "ex-api ${rate} ${filepath} ${dim_x} ${dim_y} ${dim_z} ${number_of_gpus} ${iterationsPerGpu}";
    return -1;
  }
  else if (argc > 1)
  {
    rate = std::atoi(argv[1]);
    inputFilepath = argv[2];
    nx = std::atoi(argv[3]);
    ny = std::atoi(argv[4]);
    nz = std::atoi(argv[5]);
    gpus = std::atoi(argv[6]);
    iterationsPerGpu = std::atoi(argv[7]);
  }

  fprintf(stderr, "----------ZFP-------------------\n");
  fprintf(stderr, "Parameters\n");
  fprintf(stderr, "Rate: %i\n", rate);
  fprintf(stderr, "# GPUs: %i; # iterations per GPU: %i\n", gpus, iterationsPerGpu);
  fprintf(stderr, "Input file path %s\n", inputFilepath.c_str());
  fprintf(stderr, "Dims: (%li, %li, %li)\n", nx, ny, nz);
  fprintf(stderr, "--------------------------------\n");

  int gpu = 0;
  for (int i = 0; i < gpus * iterationsPerGpu; i++)
  {
    fprintf(stderr, "Iteration #%i; GPU #%i\n", i, gpu);

    cudaSetDevice(gpu);

    size_t len = nx * ny * nz;

    Data_t *data;

    cudaMallocHost(&data->h_uncompressed_data, len * sizeof(float));
    readInputDataFromFile(inputFilepath, data->h_uncompressed_data, len);

    cudaMalloc(&data->d_uncompressed_data, len * sizeof(float));
    cudaMemcpy(data->d_uncompressed_data, data->h_uncompressed_data, len * sizeof(float), cudaMemcpyHostToDevice);

    chrono::steady_clock::time_point begin;
    chrono::steady_clock::time_point end;

    begin = std::chrono::steady_clock::now();
    NVTX_PUSH_RANGE("START_COMPRESSION_METHOD", MY_ORANGE);
    size_t compressed_data_size = compress(data, nx, ny, nz, rate);
    NVTX_POP_RANGE();
    end = std::chrono::steady_clock::now();
    fprintf(stderr, "Compression spent time %li[µs]\n", chrono::duration_cast<chrono::microseconds>(end - begin).count());
    checkIfInputIsCorrupted(data->h_uncompressed_data, data->d_uncompressed_data, len);

    begin = std::chrono::steady_clock::now();
    NVTX_PUSH_RANGE("START_DECOMPRESSION_METHOD", MY_ORANGE);
    cudaMalloc(&data->d_decompressed_data, len * sizeof(float));
    decompress(data, compressed_data_size, nx, ny, nz, rate);
    NVTX_POP_RANGE();
    end = std::chrono::steady_clock::now();
    fprintf(stderr, "DEcompression spent time %li[µs]\n", chrono::duration_cast<chrono::microseconds>(end - begin).count());

    cudaFreeHost(&data->h_decompressed_data);
    cudaMallocHost(&data->h_decompressed_data, len * sizeof(float));
    cudaMemcpy(data->h_decompressed_data, data->d_decompressed_data, len * sizeof(float), cudaMemcpyHostToDevice);
    print_error(data->h_uncompressed_data, data->h_decompressed_data, zfp_type_float, len);

    if (dumpData == true)
    {
      exportData("./decompressed-from-api", data->h_decompressed_data, sizeof(float), len);
      cudaMallocHost(&data->h_compressed_data, compressed_data_size);
      cudaMemcpy(data->h_compressed_data, data->d_compressed_data, compressed_data_size, cudaMemcpyDeviceToHost);
      exportData("./compressed-from-api", data->h_compressed_data, 1, compressed_data_size);
    }

    cudaFreeHost(data->h_decompressed_data);
    cudaFree(data->d_decompressed_data);
    cudaFreeHost(data->h_uncompressed_data);
    cudaFree(data->d_uncompressed_data);
    cudaFreeHost(data->h_compressed_data);
    cudaFree(data->d_compressed_data);

    if (gpus > 1)
    {
      if (gpu + 1 < gpus)
      {
        gpu++;
      }
      else
      {
        gpu = 0;
      }
    }

    fprintf(stderr, "\n----------------------------------------\n\n");
  }
}
