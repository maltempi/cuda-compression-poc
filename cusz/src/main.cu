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
#include <math.h>
#include "nvtx.cuh"

#include "api.hh"

#include "cli/quality_viewer.hh"
#include "cli/timerecord_viewer.hh"

using Compressor = typename cusz::Framework<float>::LorenzoFeaturedCompressor;

//#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef struct
{
    float *h_uncompressed_data;
    float *d_uncompressed_data;
    float *h_decompressed_data;
    float *d_decompressed_data;
    uint8_t *h_compressed_data;
    uint8_t *d_compressed_data;
    double eb;
    int device;
    size_t uncompressed_len;
    size_t compressed_len;
    cusz::Header header;
    char *mode;
} Data_t;

void compress(Data_t *data, size_t nx, size_t ny, size_t nz, cudaStream_t stream)
{
    size_t uncompressed_alloclen = data->uncompressed_len * 1.03;

    // Defining cusz stuff
    Compressor *compressor = new Compressor;
    cusz::TimeRecord timerecord;
    cusz::Context *ctx = new cusz::Context();
    ctx->set_len(nx, ny, nz, 1).set_eb(data->eb).set_control_string(data->mode);
    ctx->device = data->device;

    float *d_uncompressed_copy;
    cudaMalloc(&d_uncompressed_copy, sizeof(float) * nx * ny * nz);
    cudaMemcpy(d_uncompressed_copy, data->d_uncompressed_data, sizeof(float) * nx * ny * nz, cudaMemcpyHostToDevice);

    cusz::Context::adjust_eb(ctx, d_uncompressed_copy);

    NVTX_PUSH_RANGE("CUSZ_COMPRESS", MY_YELLOW);
    cusz::core_compress(compressor, ctx,                                             // compressor & config
                        d_uncompressed_copy, uncompressed_alloclen,                  // input
                        data->d_compressed_data, data->compressed_len, data->header, // output
                        stream, &timerecord);
    NVTX_POP_RANGE();

    cudaFree(d_uncompressed_copy);
    delete compressor;
}

void decompress(Data_t *data, cudaStream_t stream)
{
    auto compressor = new Compressor;
    cusz::TimeRecord timerecord;
    size_t uncompressed_alloclen = data->uncompressed_len * 1.03;

    cudaMalloc(&data->d_uncompressed_data, sizeof(float) * uncompressed_alloclen);

    NVTX_PUSH_RANGE("CUSZ_DECOMPRESS", MY_YELLOW);
    cusz::core_decompress(compressor, &data->header,
                          data->d_compressed_data,   // input
                          data->compressed_len,      // input len
                          data->d_decompressed_data, // output
                          uncompressed_alloclen,     // output len
                          stream, &timerecord);
    NVTX_POP_RANGE();

    // cusz::TimeRecordViewer::view_decompression(&timerecord, len * sizeof(float));

    delete compressor;
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
    double eb = 1e-4;
    char *mode = "mode=abs"; // "abs" or "r2r"
    string inputFilepath = "../../hurr-CLOUDf48-500x500x100";
    size_t nx = 500;
    size_t ny = 500;
    size_t nz = 100;
    bool dumpData = false;
    bool printReport = false;

    fprintf(stderr, "----------CUSZ------------------\n");
    fprintf(stderr, "Parameters\n");
    fprintf(stderr, "EB: %lf; Mode: %s\n", eb, mode);
    fprintf(stderr, "# GPUs: %i; # iterations per GPU: %i\n", gpus, iterationsPerGpu);
    fprintf(stderr, "Input file path %s\n", inputFilepath.c_str());
    fprintf(stderr, "Dims: (%li, %li, %li)\n", nx, ny, nz);
    fprintf(stderr, "--------------------------------\n");

    int gpu = 0;
    for (int i = 0; i < gpus * iterationsPerGpu; i++)
    {
        fprintf(stderr, "Iteration #%i; GPU #%i\n", i, gpu);

        cudaSetDevice(gpu);

        Data_t _data;
        Data_t *data = &_data;
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        size_t len = nx * ny * nz;
        data->uncompressed_len = len;
        data->eb = eb;
        data->mode = mode;
        data->device = gpu;

        cudaMallocHost(&data->h_uncompressed_data, len * sizeof(float));
        readInputDataFromFile(inputFilepath, data->h_uncompressed_data, len);

        cudaMalloc(&data->d_uncompressed_data, len * sizeof(float));
        cudaMemcpy(data->d_uncompressed_data, data->h_uncompressed_data, len * sizeof(float), cudaMemcpyHostToDevice);

        chrono::steady_clock::time_point begin;
        chrono::steady_clock::time_point end;

        begin = std::chrono::steady_clock::now();
        NVTX_PUSH_RANGE("START_COMPRESSION_METHOD", MY_ORANGE);
        compress(data, nx, ny, nz, stream);
        NVTX_POP_RANGE();
        end = std::chrono::steady_clock::now();
        fprintf(stderr, "Compression spent time %li[µs]\n", chrono::duration_cast<chrono::microseconds>(end - begin).count());

        fprintf(stderr, "Starting decompression\n");
        begin = std::chrono::steady_clock::now();
        NVTX_PUSH_RANGE("START_DECOMPRESSION_METHOD", MY_ORANGE);
        cudaMalloc(&data->d_decompressed_data, len * sizeof(float));
        decompress(data, stream);
        NVTX_POP_RANGE();
        end = std::chrono::steady_clock::now();
        fprintf(stderr, "DEcompression spent time %li[µs]\n", chrono::duration_cast<chrono::microseconds>(end - begin).count());

        if (dumpData)
        {
            exportData("./dump/decompressed-from-api_" + std::to_string(i), data->h_decompressed_data, sizeof(float), len);
            cudaMallocHost(&data->h_compressed_data, data->compressed_len);
            cudaMemcpy(data->h_compressed_data, data->d_compressed_data, data->compressed_len, cudaMemcpyDeviceToHost);
            exportData("./dump/compressed-from-api_" + std::to_string(i), data->h_compressed_data, 1, data->compressed_len);
        }

        if (printReport)
        {
            fprintf(stderr, "Report:\n");
            cudaFreeHost(&data->h_decompressed_data);
            cudaMallocHost(&data->h_decompressed_data, len * sizeof(float));
            cudaMemcpy(data->h_decompressed_data, data->d_decompressed_data, len * sizeof(float), cudaMemcpyHostToDevice);

            fprintf(stderr, "CPU Metrics:\n");
            cusz::QualityViewer::echo_metric_cpu(data->h_decompressed_data, data->h_uncompressed_data, len, size_t(data->compressed_len), false);
        }

        cudaFreeHost(&data->h_decompressed_data);
        cudaFree(data->d_decompressed_data);
        cudaFreeHost(&data->h_uncompressed_data);
        cudaFree(data->d_uncompressed_data);
        cudaFreeHost(&data->h_compressed_data);
        cudaFree(data->d_compressed_data);
        cudaStreamDestroy(stream);

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
