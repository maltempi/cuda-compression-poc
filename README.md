# An analysis of cuSZ and cuZFP

Thiago J M Maltempi
[tmmaltempi@gmail.com](mailto:tmmaltempi@gmail.com)
November 2022


Experiment description
======================

This experiment is intended to be a comparison between two scientific-data compression libraries on GPU devices, cuSZ (actual: cuSZ+) and ZFP (actual name: cuZFP) in terms of quality, memory footprint and how fast they are.

Those target libraries use different (de)compression modes: cuSZ is an error-bounded lossy compressor while ZFP supports only fixed-rate mode\[2\]. Knowing that, this experiment defines four different tests scenarios, running on 4 GPUs, 10 times each: 1) cuSZ in absolute mode (cusz\_abs\_4gpu\_10iter-each); 2) cuSZ in r2r mode (cusz\_r2r\_4gpu\_10iter-each); 3) ZFP using rate=4; 4) ZFP using rate=8.

Source code
===========

Available in a private repository: [https://github.com/maltempi/cuda-compression-poc](https://www.google.com/url?q=https://github.com/maltempi/cuda-compression-poc&sa=D&source=editors&ust=1668530229111241&usg=AOvVaw3r0iZ5yjWKcv0O52GxGbbw)

Each test scenario has its own git tag:

*   [zfp\_rate8\_4gpu\_10iter-each](https://www.google.com/url?q=https://github.com/maltempi/cuda-compression-poc/tree/zfp_rate8_4gpu_10iter-each&sa=D&source=editors&ust=1668530229111762&usg=AOvVaw2VTpuc1rzbxc90dwqe-UjJ)
*   [zfp\_rate4\_4gpu\_10iter-each](https://www.google.com/url?q=https://github.com/maltempi/cuda-compression-poc/tree/zfp_rate4_4gpu_10iter-each&sa=D&source=editors&ust=1668530229112067&usg=AOvVaw0APHQcjsjeIHk_MDltqRjl)
*   [cusz\_r2r\_4gpu\_10iter-each](https://www.google.com/url?q=https://github.com/maltempi/cuda-compression-poc/tree/cusz_r2r_4gpu_10iter-each&sa=D&source=editors&ust=1668530229112637&usg=AOvVaw2kmtqJv1rQXXKQO24RQScP)
*   [cusz\_abs\_4gpu\_10iter-each](https://www.google.com/url?q=https://github.com/maltempi/cuda-compression-poc/tree/cusz_abs_4gpu_10iter-each&sa=D&source=editors&ust=1668530229112925&usg=AOvVaw1hXiQ-iwndXih5cg6cxrg1)

Dataset
=======

Name: Hurricane dataset

Dimensions: 500x500x100

Available at: [IEEE Visualization 2004 Contest: Data Set (computer.org)](https://www.google.com/url?q=http://vis.computer.org/vis2004contest/data.html&sa=D&source=editors&ust=1668530229113679&usg=AOvVaw0Wm1GW51hfXffERWUsv9VE)

Size: 500\*500\*100 \* sizeof(float) = 100,000,000 bytes or 97,656KB or 95.36MB

Environment
===========

Cluster: HPC Ogbon

GPU: 4 x Tesla V100-SXM2-32GB

Container: maltempi/awave-dev:ubuntu20.04-cuda11.2-customcusz-zfp

Container Hash: [DIGEST:sha256:785cc1dfe1efa5ccdcf55e20a5a9a053](https://www.google.com/url?q=https://hub.docker.com/layers/maltempi/awave-dev/ubuntu20.04-cuda11.2-customcusz-zfp/images/sha256-785cc1dfe1efa5ccdcf55e20a5a9a053d05e000fad35525e93c6bc6ebe0ff693?context%3Dexplore&sa=D&source=editors&ust=1668530229115075&usg=AOvVaw1GuH9S_TQxDxtipRjkAQvX)

cuSZ source code: [maltempi/cuSZ at c60111c5f69b6e50f9906299bfd5c06d99381325 (github.com)](https://www.google.com/url?q=https://github.com/maltempi/cuSZ/tree/c60111c5f69b6e50f9906299bfd5c06d99381325&sa=D&source=editors&ust=1668530229115698&usg=AOvVaw2Z3E4YjC7Q0m03ovs2kbWu)

ZFP source code: [LLNL/zfp at f39af72648a2aeb88e9b2cca8c64f51b493ad5f4 (github.com)](https://www.google.com/url?q=https://github.com/LLNL/zfp/tree/f39af72648a2aeb88e9b2cca8c64f51b493ad5f4&sa=D&source=editors&ust=1668530229116202&usg=AOvVaw1TPw-2e3P4YScKp9HAd38o)

Considerations
==============

During nsight report analysis, it has noticed that the first compression execution had long execution times. Considering it a possible warming-up of the system, this report is ignoring the first execution of all test scenarios. The slower executions can be easily found on nsight, as illustrated on the figure below.

![](images/image2.png)

Another important consideration, cuSZ contains some code changes for our scenarios in order to support running on multiple devices.

Highlights
==========

*   The fastest compression: ZFP (79 microsec; 74 microsec)
*   The fastest decompression: ZFP (16 microsec; 18 microsec)
*   The best recovery quality (PSNR): cuSZ r2r (94.18)
*   The best compression ratio: cuSZ r2r (29.24)
*   cuSZ corrupts the input data in GPU, a copy is required before submitting the data to compression. More processing time and memory are required. \[3\]
*   cuSZ tests indicate memory leak (see memory footprint section in this document). In the end of each test scenario cuSZ left a residue of ~1.5G.

Test scenario

Compression

spent time (AVG)

(microsec)

Compression spent time

(STD)

(microsec)

Decompression

spent time

(AVG)

(microsec)

Decompress spent time

(STD)

(microsec)

PSNR

Comp ratio

cusz abs 1e-4

6537

343781

4773

108

43.55

29.24

cusz r2r 1e-4

6714

347145

5896

112

94.18

16.27

zfp rate=4

79

441364

16

2

49.71

8

zfp rate=8

74

443058

18

8

72.95

4

Note: All measured spent times are considered discarding the first compression and decompression. Values are very similar with a median.

Memory footprint
================

The following table shows the expectation of memory usage during compression and decompression for each test scenario and how much memory they have actually taken during the first execution. The “Residue” field refers to how much memory is still allocated after the execution. It may suggest a memory leak.

Test scenario

During Compression

(expectation)

(MB)

During decompression

(expectation)

(MB)

During compression

(actual)

(MB)

During decompression

(actual)

(MB)

Residue

(MB)

cuSZ abs

193.92

193.92

500.93

646.92

143.13

cusz r2r

193.92

193.92

500.93

646.92

143.13

zfp rate=4

107.36

202.72

107.29

202.66

0

zfp rate=8

119.36

214.72

119.21

214.58

0

Investigating the reasons of “residue”, it’s possible to notice that cuSZ allocates more and more memory for each iteration on a device, as illustrated on the chart below.

![](images/image3.png)

While ZFP frees all data allocated for the current iteration, as illustrated below.

![](images/image1.png)

### Details of the memory usage expectation:

Test scenario

During compression

During decompression

cuSZ abs

*   Uncompressed data = 95.36MB
*   Uncompressed data copy = 95.36MB
*   Compressed data = 3.2MB

Total = 193.92MB

*   Uncompressed data = 95.36MB
*   Compressed data = 3.2MB
*   Decompressed data = 95.36MB

Total = 193.92MB

cuSZ r2r

*   Uncompressed data = 95.36MB
*   Uncompressed data copy = 95.36MB
*   Compressed data = 5.8MB

Total: 196.52MB

*   Uncompressed data = 95.36MB
*   Compressed data = 5.8MB
*   Decompressed data = 95.36MB

Total: 196.52MB

ZFP rate=4

*   Uncompressed data = 95.36MB
*   Compressed data = 12MB

Total: 107.36MB

*   Uncompressed data = 95.36MB
*   Compressed data = 12MB
*   Decompressed data = 95.36MB

Total: 202.72

ZFP rate = 8

*   Uncompressed data = 95.36MB
*   Compressed data = 24MB

Total: 119.36 MB

*   Uncompressed data = 95.36MB
*   Compressed data = 24MB
*   Decompressed data = 95.36MB

Total: 214.72 MB

References
==========

\[1\] [szcompressor/cuSZ: A GPU accelerated error-bounded lossy compression for scientific data. (github.com)](https://www.google.com/url?q=https://github.com/szcompressor/cuSZ/&sa=D&source=editors&ust=1668530229149609&usg=AOvVaw2zdBbrxo9dbOW8h3zc6hEx)

\[2\] [https://zfp.readthedocs.io/en/release1.0.0/execution.html#execution-parameters](https://www.google.com/url?q=https://zfp.readthedocs.io/en/release1.0.0/execution.html%23execution-parameters&sa=D&source=editors&ust=1668530229150108&usg=AOvVaw0m53S0fPUs4cWTVl6nCtNr)

\[3\] [(question) cusz changes input data after compression · Issue #70 · szcompressor/cuSZ (github.com)](https://www.google.com/url?q=https://github.com/szcompressor/cuSZ/issues/70&sa=D&source=editors&ust=1668530229150475&usg=AOvVaw0QO5bC9GbgqJaa94v724JL)
