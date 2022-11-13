/**
 * @file ex_api_core.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-10
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "api.hh"

#include "cli/quality_viewer.hh"
#include "cli/timerecord_viewer.hh"

#include "nvtx.cuh"

template <typename T>
void f(std::string fname)
{
    using Compressor = typename cusz::Framework<T>::LorenzoFeaturedCompressor;

    /*
     * The predefined XFeatureCompressor is equivalent to the free form below.
     *
     * using Framework   = cusz::Framework<T>;
     * using Combination = cusz::CompressorTemplate<
     *     typename Framework::PredictorLorenzo,
     *     typename Framework::SpCodecCSR,
     *     typename Framework::CodecHuffman32,
     *     typename Framework::CodecHuffman64>;
     * using Compressor = cusz::Compressor<Combination>;
     */

    /* For demo, we use 3600x1800 CESM data. */
    auto len = 500 * 500 * 100;

    Compressor*  compressor;
    cusz::Header header;
    BYTE*        compressed;
    size_t       compressed_len;

    T *d_uncompressed, *h_uncompressed;
    T *d_decompressed, *h_decompressed;

    /* cuSZ requires a 3% overhead on device (not required on host). */
    size_t uncompressed_alloclen = len * 1.03;
    size_t decompressed_alloclen = uncompressed_alloclen;

    /* code snippet for looking at the device array easily */
    auto peek_devdata = [](T* d_arr, size_t num = 20) {
        thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const T i) { printf("%f\t", i); });
        printf("\n");
    };

    // clang-format off
    cudaMalloc(     &d_uncompressed, sizeof(T) * uncompressed_alloclen );
    cudaMallocHost( &h_uncompressed, sizeof(T) * len );
    cudaMalloc(     &d_decompressed, sizeof(T) * decompressed_alloclen );
    cudaMallocHost( &h_decompressed, sizeof(T) * len );
    // clang-format on

    /* User handles loading from filesystem & transferring to device. */
    io::read_binary_to_array(fname, h_uncompressed, len);
    cudaMemcpy(d_uncompressed, h_uncompressed, sizeof(T) * len, cudaMemcpyHostToDevice);

    /* a casual peek */
    printf("peeking uncompressed data, 20 elements\n");
    peek_devdata(d_uncompressed, 20);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    compressor = new Compressor;
    BYTE* exposed_compressed;
    {
        NVTX_PUSH_RANGE("START_COMPRESSION_METHOD", MY_ORANGE);
        cusz::TimeRecord timerecord;
        cusz::Context*   ctx;

        /*
         * Two methods to build the configuration.
         * Note that specifying type is not needed because of T in cusz::Framework<T>
         */

        /* Method 1: Everthing is in string. */
        // char const* config = "eb=2.4e-4,mode=r2r,len=3600x1800";
        // ctx                = new cusz::Context(config);

        /* Method 2: Numeric and string options are set separatedly. */
        ctx = new cusz::Context();
        ctx->set_len(500, 500, 100, 1)        // In this case, the last 2 arguments can be omitted.
            .set_eb(1e-4)                   // numeric
            .set_control_string("mode=r2r");  // string

        float *d_uncompressed_copy;
        cudaMalloc(&d_uncompressed_copy, sizeof(float) * len);
        cudaMemcpy(d_uncompressed_copy, d_uncompressed, sizeof(float) * len, cudaMemcpyHostToDevice);

        cout << "EB (before adjust_eb): " << ctx->eb << "\n";
        cusz::Context::adjust_eb(ctx, d_uncompressed);
        cout << "EB (after adjust_eb): " << ctx->eb << "\n";

        NVTX_PUSH_RANGE("CUSZ_COMPRESS", MY_YELLOW);
        cusz::core_compress(
            compressor, ctx,                             // compressor & config
            d_uncompressed_copy, uncompressed_alloclen,       // input
            exposed_compressed, compressed_len, header,  // output
            stream, &timerecord);
        NVTX_POP_RANGE();

        cudaFree(&d_uncompressed_copy);
        NVTX_POP_RANGE();

        /* User can interpret the collected time information in other ways. */
        cusz::TimeRecordViewer::view_compression(&timerecord, len * sizeof(T), compressed_len);

        /* verify header */
        // clang-format off
        printf("header.%-*s : %x\n",            12, "(addr)", &header);
        printf("header.%-*s : %lu, %lu, %lu\n", 12, "{x,y,z}", header.x, header.y, header.z);
        printf("header.%-*s : %lu\n",           12, "filesize", header.get_filesize());
        // clang-format on
    }

    /* If needed, User should perform a memcopy to transfer `exposed_compressed` before `compressor` is destroyed. */
    cudaMalloc(&compressed, compressed_len);
    cudaMemcpy(compressed, exposed_compressed, compressed_len, cudaMemcpyDeviceToDevice);

    /* release compressor */ delete compressor;

    compressor = new Compressor;
    {
        NVTX_PUSH_RANGE("START_DECOMPRESSION_METHOD", MY_ORANGE);
        cusz::TimeRecord timerecord;

        NVTX_PUSH_RANGE("CUSZ_DECOMPRESS", MY_YELLOW);
        cusz::core_decompress(
            compressor, &header,                    // compressor & config
            compressed, compressed_len,             // input
            d_decompressed, decompressed_alloclen,  // output
            stream, &timerecord);
        NVTX_POP_RANGE();
        NVTX_POP_RANGE();

        /* User can interpret the collected time information in other ways. */
        cusz::TimeRecordViewer::view_decompression(&timerecord, len * sizeof(T));
    }

    /* a casual peek */
    printf("peeking decompressed data, 20 elements\n");
    peek_devdata(d_decompressed, 20);

    /* demo: offline checking (de)compression quality. */
    /* load data again    */ cudaMemcpy(d_uncompressed, h_uncompressed, sizeof(T) * len, cudaMemcpyHostToDevice);
    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(d_decompressed, d_uncompressed, len, compressed_len);

    cudaFree(compressed);
    cudaFree(d_uncompressed);
    cudaFree(h_decompressed);
    cudaFree(h_decompressed);
    cudaFree(h_uncompressed);
    cudaFree(exposed_compressed);

    delete compressor;

    cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
    for (int i = 0 ; i < 3; i++) {
        f<float>("/home/thiago.maltempi/workspace/cuda-compression-poc/hurr-CLOUDf48-500x500x100");
    }
    return 0;
}

