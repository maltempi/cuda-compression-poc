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
    size_t uncompressed_len;
    size_t compressed_len;
    cusz::Header header;
    char* mode;
} Data_t;

void compress(Data_t *data, size_t nx, size_t ny, size_t nz, cudaStream_t stream)
{
    size_t uncompressed_alloclen = data->uncompressed_len * 1.03;

    // Defining cusz stuff
    Compressor *compressor = new Compressor;
    cusz::TimeRecord timerecord;
    cusz::Context *ctx = new cusz::Context();
    ctx->set_len(nx, ny, nz, 1).set_eb(data->eb).set_control_string(data->mode);
    ctx->device = 0;

    cusz::core_compress(compressor, ctx,                                             // compressor & config
                        data->d_uncompressed_data, uncompressed_alloclen,            // input
                        data->d_compressed_data, data->compressed_len, data->header, // output
                        stream, &timerecord);

    delete compressor;
    delete ctx;
}

void decompress(Data_t *data, cudaStream_t stream)
{
    auto compressor = new Compressor;
    cusz::TimeRecord timerecord;
    size_t uncompressed_alloclen = data->uncompressed_len * 1.03;

    cudaFree(data->d_decompressed_data);
    cudaMalloc(&data->d_uncompressed_data, sizeof(float) * uncompressed_alloclen);

    cusz::core_decompress(compressor, &data->header,
                          data->d_compressed_data,   // input
                          data->compressed_len,      // input len
                          data->d_decompressed_data, // output
                          uncompressed_alloclen,     // output len
                          stream, &timerecord);

    // cusz::TimeRecordViewer::view_decompression(&timerecord, len * sizeof(float));

    delete compressor;
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
        cout << errors << " errors happened in the input data after compression\n";
    }
    else
    {
        cout << "No errors have been found in the input data after compression.\n";
    }

    cudaFreeHost(&h_input_after_compression);

    return 0;
}

static void
print_error(const void *fin, const void *fout, size_t n)
{
    // Code from: https://github.com/LLNL/zfp/blob/fc96c9158e8befc227f9f46093f5e1b35723be02/utils/zfp.c#L30-L81
    const float *f32i = (const float *)fin;
    const float *f32o = (const float *)fout;
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
        d = fabs((double)(f32i[i] - f32o[i]));
        val = (double)f32i[i];
        emax = MAX(emax, d);
        erms += d * d;
        fmin = MIN(fmin, val);
        fmax = MAX(fmax, val);
    }
    erms = sqrt(erms / n);
    ermsn = erms / (fmax - fmin);
    psnr = 20 * log10((fmax - fmin) / (2 * erms));
    fprintf(stderr, "-> Stats = rmse=%.4g nrmse=%.4g maxe=%.4g psnr=%.2f\n\n", erms, ermsn, emax, psnr);
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
    cout << "Dumping data in " << path << "\n";
    auto file = fopen(path.c_str(), "wb");
    fwrite(h_data, data_size, len, file);
    fclose(file);
}

int main(int argc, char *argv[])
{
    Data_t _data;
    Data_t *data = &_data;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* allocate array of floats */
    size_t nx = 500;
    size_t ny = 500;
    size_t nz = 100;
    size_t len = nx * ny * nz;
    string inputFilepath = "/opt/zfp/bin/hurr-CLOUDf48-500x500x100";
    bool dumpData = true;
    data->uncompressed_len = len;
    data->eb = 1e-4;
    data->mode = "mode=r2r"; // or "abs"


    cout << "----------------------------------------------------------------------\n";
    cout << "Using CUSZ for compression.\n";
    cout << "----------------------------------------------------------------------\n";
    cout << "nx=" << nx << " ny=" << ny << " nz=" << nz << " total=" << len << "\n";
    cout << "eb=" << std::scientific << data->eb << "\n";
    cout << data->mode << "\n";
    cout << "compressing " << inputFilepath << "\n";
    cout << "dump data " << inputFilepath << "\n";
    cout << "----------------------------------------------------------------------\n";

    cudaMallocHost(&data->h_uncompressed_data, len * sizeof(float));
    readInputDataFromFile(inputFilepath, data->h_uncompressed_data, len);

    cudaMalloc(&data->d_uncompressed_data, len * sizeof(float));
    cudaMemcpy(data->d_uncompressed_data, data->h_uncompressed_data, len * sizeof(float), cudaMemcpyHostToDevice);

    chrono::steady_clock::time_point begin;
    chrono::steady_clock::time_point end;

    cout << "Starting Compression\n";
    begin = std::chrono::steady_clock::now();
    NVTX_PUSH_RANGE("START_COMPRESSION_METHOD", MY_ORANGE);
    compress(data, nx, ny, nz, stream);
    NVTX_POP_RANGE();
    end = std::chrono::steady_clock::now();
    cout << "Compression spent time: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]\n";
    cout << "----------------------------------------------------------------------\n";
    checkIfInputIsCorrupted(data->h_uncompressed_data, data->d_uncompressed_data, len);

    cout << "Starting decompression\n";
    begin = std::chrono::steady_clock::now();
    NVTX_PUSH_RANGE("START_DECOMPRESSION_METHOD", MY_ORANGE);
    cudaMalloc(&data->d_decompressed_data, len * sizeof(float));
    decompress(data, stream);
    NVTX_POP_RANGE();
    end = std::chrono::steady_clock::now();
    cout << "Decompression spent time: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]\n";
    cout << "----------------------------------------------------------------------\n";

    cout << "Report:\n";
    cudaFreeHost(&data->h_decompressed_data);
    cudaMallocHost(&data->h_decompressed_data, len * sizeof(float));
    cudaMemcpy(data->h_decompressed_data, data->d_decompressed_data, len * sizeof(float), cudaMemcpyHostToDevice);
    print_error(data->h_uncompressed_data, data->h_decompressed_data, len);

    if (dumpData == true)
    {
        exportData("./decompressed-from-api", data->h_decompressed_data, sizeof(float), len);
        cudaMallocHost(&data->h_compressed_data, data->compressed_len);
        cudaMemcpy(data->h_compressed_data, data->d_compressed_data, data->compressed_len, cudaMemcpyDeviceToHost);
        exportData("./compressed-from-api", data->h_compressed_data, 1, data->compressed_len);
    }

    cudaFreeHost(&data->h_decompressed_data);
    cudaFree(&data->d_decompressed_data);
    cudaFreeHost(&data->h_uncompressed_data);
    cudaFree(&data->d_uncompressed_data);
    cudaFreeHost(&data->h_compressed_data);
    cudaFree(&data->d_compressed_data);

    cudaStreamDestroy(stream);
}
