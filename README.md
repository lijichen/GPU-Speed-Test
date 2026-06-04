gpu_stress_nccl usage:
<img width="1028" height="390" alt="image" src="https://github.com/user-attachments/assets/cf43dbc1-22f2-4f29-803f-fb45e60e7500" />
<br>
No specific the arh:
<br>
nvcc -O3 -std=c++17 gpu_stress_nccl_g2g_link.cu -o gpu_stress_nccl -ldl -Xcompiler -pthread
<br>
H100/800:
<br>
nvcc -O3 -std=c++17   -gencode=arch=compute_90,code=sm_90   gpu_stress_nccl_g2g_link.cu   -o gpu_stress_nccl   -ldl -Xcompiler -pthread

```Bash
The example:
# ./gpu_stress_nccl
CUDA GPUs: 8
Single-GPU memory stress duration: 30 sec
All-GPU memory/NCCL stress duration: 30 sec (same as single-GPU)
Memory reserve per GPU: 0.50 GiB
GPU-to-GPU pair test: enabled, per method/directed pair=1 sec, buffer=0.50 GiB, batch=4, host-staged=fallback-only, bidir-pair=disabled
GPU-to-GPU aggregate test: enabled, per focus GPU/mode=1 sec
Temperature monitor: enabled, threshold=80.0 C, interval=1000 ms
NCCL: available, version code 23004

Detected GPU memory:
  GPU 0  NVIDIA A40  total=44.42 GiB
  GPU 1  NVIDIA A40  total=44.42 GiB
  GPU 2  NVIDIA A40  total=44.42 GiB
  GPU 3  NVIDIA A40  total=44.42 GiB
  GPU 4  NVIDIA A40  total=44.42 GiB
  GPU 5  NVIDIA A40  total=44.42 GiB
  GPU 6  NVIDIA A40  total=44.42 GiB
  GPU 7  NVIDIA A40  total=44.42 GiB

GPU-to-GPU topology links from nvidia-smi topo -m plus CUDA P2P fallback for unknown pairs:
 src\dst      GPU0      GPU1      GPU2      GPU3      GPU4      GPU5      GPU6      GPU7
    GPU0         X       PXB       PXB       PXB       SYS       SYS       SYS       SYS
    GPU1  CUDA_P2P         X       PXB       PXB       SYS       SYS       SYS       SYS
    GPU2  CUDA_P2P       PXB         X       PIX       SYS       SYS       SYS       SYS
    GPU3  CUDA_P2P       PXB       PIX         X       SYS       SYS       SYS       SYS
    GPU4  CUDA_P2P       SYS       SYS       SYS         X       PXB       PXB       PXB
    GPU5  CUDA_P2P       SYS       SYS       SYS       PXB         X       PXB       PXB
    GPU6  CUDA_P2P       SYS       SYS       SYS       PXB       PXB         X       PIX
    GPU7  CUDA_P2P       SYS       SYS       SYS       PXB       PXB       PIX         X
Link legend: NV#=NVLink/NVSwitch bonded NVLinks, PIX/PXB/PHB/NODE/SYS=PCIe paths with increasing distance.
Temperature monitor: NVML ready; warnings will be printed when a GPU reaches 80.0 C or higher.

Sequential memory stress, one GPU at a time...
  Testing GPU 0 for 30.0 sec... done, 550.19 GB/s
  Testing GPU 1 for 30.00 sec... done, 550.23 GB/s
  Testing GPU 2 for 30.00 sec... done, 550.23 GB/s
  Testing GPU 3 for 30.00 sec... done, 550.21 GB/s
  Testing GPU 4 for 30.00 sec... done, 550.26 GB/s
  Testing GPU 5 for 30.00 sec... done, 550.24 GB/s
  Testing GPU 6 for 30.00 sec... done, 550.21 GB/s
  Testing GPU 7 for 30.00 sec... done, 550.20 GB/s

Concurrent memory stress, all GPUs together for 30.00 sec...
  All-GPU total memory stress bandwidth: 4401.75 GB/s

NCCL AllReduce link test, all GPUs together for 30.00 sec...
  NCCL buffer per GPU: 0.25 GiB
  NCCL iterations: 232, elapsed=30.428 sec
  NCCL AllReduce algBW: 2.05 GB/s
  NCCL AllReduce busBW/link estimate: 3.58 GB/s

GPU-to-GPU pair bandwidth test, all directed GPU pairs, 1.00 sec per method, buffer=0.50 GiB...
  Methods: cudaMemcpyPeer, peer remote read, peer remote write, pinned host-staged copy, NCCL Send/Recv when supported.
  NCCL Send/Recv pair test: enabled.
  Testing GPU 0 -> GPU 1 [PXB] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PXB
  Testing GPU 0 -> GPU 2 [PXB] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PXB
  Testing GPU 0 -> GPU 3 [PXB] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PXB
  Testing GPU 0 -> GPU 4 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.96 GB/s via SYS
  Testing GPU 0 -> GPU 5 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.85 GB/s via SYS
  Testing GPU 0 -> GPU 6 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.04 GB/s via SYS
  Testing GPU 0 -> GPU 7 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.96 GB/s via SYS
  Testing GPU 1 -> GPU 0 [CUDA_P2P] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via CUDA_P2P
  Testing GPU 1 -> GPU 2 [PXB] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PXB
  Testing GPU 1 -> GPU 3 [PXB] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PXB
  Testing GPU 1 -> GPU 4 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.97 GB/s via SYS
  Testing GPU 1 -> GPU 5 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.96 GB/s via SYS
  Testing GPU 1 -> GPU 6 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.12 GB/s via SYS
  Testing GPU 1 -> GPU 7 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.06 GB/s via SYS
  Testing GPU 2 -> GPU 0 [CUDA_P2P] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via CUDA_P2P
  Testing GPU 2 -> GPU 1 [PXB] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PXB
  Testing GPU 2 -> GPU 3 [PIX] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PIX
  Testing GPU 2 -> GPU 4 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.02 GB/s via SYS
  Testing GPU 2 -> GPU 5 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.07 GB/s via SYS
  Testing GPU 2 -> GPU 6 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.96 GB/s via SYS
  Testing GPU 2 -> GPU 7 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.93 GB/s via SYS
  Testing GPU 3 -> GPU 0 [CUDA_P2P] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via CUDA_P2P
  Testing GPU 3 -> GPU 1 [PXB] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PXB
  Testing GPU 3 -> GPU 2 [PIX] for 1.00 sec per method... best CUDA_COPY 20.68 GB/s via PIX
  Testing GPU 3 -> GPU 4 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.03 GB/s via SYS
  Testing GPU 3 -> GPU 5 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.94 GB/s via SYS
  Testing GPU 3 -> GPU 6 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.15 GB/s via SYS
  Testing GPU 3 -> GPU 7 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 18.99 GB/s via SYS
  Testing GPU 4 -> GPU 0 [CUDA_P2P] for 1.00 sec per method... best NCCL_SENDRECV 19.13 GB/s via CUDA_P2P
  Testing GPU 4 -> GPU 1 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.24 GB/s via SYS
  Testing GPU 4 -> GPU 2 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.29 GB/s via SYS
  Testing GPU 4 -> GPU 3 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.07 GB/s via SYS
  Testing GPU 4 -> GPU 5 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 4 -> GPU 6 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 4 -> GPU 7 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 5 -> GPU 0 [CUDA_P2P] for 1.00 sec per method... best NCCL_SENDRECV 19.24 GB/s via CUDA_P2P
  Testing GPU 5 -> GPU 1 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.42 GB/s via SYS
  Testing GPU 5 -> GPU 2 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.59 GB/s via SYS
  Testing GPU 5 -> GPU 3 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.18 GB/s via SYS
  Testing GPU 5 -> GPU 4 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 5 -> GPU 6 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 5 -> GPU 7 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 6 -> GPU 0 [CUDA_P2P] for 1.00 sec per method... best NCCL_SENDRECV 19.32 GB/s via CUDA_P2P
  Testing GPU 6 -> GPU 1 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.36 GB/s via SYS
  Testing GPU 6 -> GPU 2 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.19 GB/s via SYS
  Testing GPU 6 -> GPU 3 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.25 GB/s via SYS
  Testing GPU 6 -> GPU 4 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 6 -> GPU 5 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 6 -> GPU 7 [PIX] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PIX
  Testing GPU 7 -> GPU 0 [CUDA_P2P] for 1.00 sec per method... best NCCL_SENDRECV 19.44 GB/s via CUDA_P2P
  Testing GPU 7 -> GPU 1 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.25 GB/s via SYS
  Testing GPU 7 -> GPU 2 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.45 GB/s via SYS
  Testing GPU 7 -> GPU 3 [SYS] for 1.00 sec per method... best NCCL_SENDRECV 19.01 GB/s via SYS
  Testing GPU 7 -> GPU 4 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 7 -> GPU 5 [PXB] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PXB
  Testing GPU 7 -> GPU 6 [PIX] for 1.00 sec per method... best CUDA_COPY 20.69 GB/s via PIX

GPU-to-GPU aggregate bandwidth test, one focus GPU at a time, 1.00 sec per aggregate mode, slice=0.50 GiB...
  Modes: one-to-all, all-to-one, bidirectional aggregate cudaMemcpyPeer.
  Aggregate GPU 0 [PXBx3,SYSx4] for 1.00 sec per aggregate mode... bidir 23.05 GB/s
  Aggregate GPU 1 [CUDA_P2Px1,PXBx2,SYSx4] for 1.00 sec per aggregate mode... bidir 23.57 GB/s
  Aggregate GPU 2 [CUDA_P2Px1,PIXx1,PXBx1,SYSx4] for 1.00 sec per aggregate mode... bidir 22.82 GB/s
  Aggregate GPU 3 [CUDA_P2Px1,PIXx1,PXBx1,SYSx4] for 1.00 sec per aggregate mode... bidir 23.98 GB/s
  Aggregate GPU 4 [CUDA_P2Px1,PXBx3,SYSx3] for 1.00 sec per aggregate mode... bidir 18.69 GB/s
  Aggregate GPU 5 [CUDA_P2Px1,PXBx3,SYSx3] for 1.00 sec per aggregate mode... bidir 17.95 GB/s
  Aggregate GPU 6 [CUDA_P2Px1,PIXx1,PXBx2,SYSx3] for 1.00 sec per aggregate mode... bidir 19.12 GB/s
  Aggregate GPU 7 [CUDA_P2Px1,PIXx1,PXBx2,SYSx3] for 1.00 sec per aggregate mode... bidir 20.13 GB/s

================ Summary ================
GPU  Name                                    VRAM         Alloc      SeqMemGB/s      AllMemGB/s   MaxTempC   TempAlert
----------------------------------------------------------------------------------------------------------------------
0    NVIDIA A40                         44.42 GiB     43.66 GiB          550.19          550.20       59 C          NO
1    NVIDIA A40                         44.42 GiB     43.66 GiB          550.23          550.25       62 C          NO
2    NVIDIA A40                         44.42 GiB     43.66 GiB          550.23          550.22       59 C          NO
3    NVIDIA A40                         44.42 GiB     43.66 GiB          550.21          550.23       61 C          NO
4    NVIDIA A40                         44.42 GiB     43.66 GiB          550.26          550.22       57 C          NO
5    NVIDIA A40                         44.42 GiB     43.66 GiB          550.24          550.20       63 C          NO
6    NVIDIA A40                         44.42 GiB     43.66 GiB          550.21          550.23       63 C          NO
7    NVIDIA A40                         44.42 GiB     43.66 GiB          550.20          550.20       65 C          NO
----------------------------------------------------------------------------------------------------------------------
Single-GPU test duration: 30.00 sec
All-GPU test duration: 30.00 sec (same as single-GPU)
All-GPU memory total: 4401.75 GB/s
GPU-to-GPU pair test duration per method/directed pair: 1.00 sec
GPU-to-GPU pair buffer: 0.50 GiB
GPU-to-GPU aggregate test duration per focus GPU/mode: 1.00 sec
Temperature threshold: 80.0 C; alert triggered: NO

================ GPU-to-GPU Pair Summary ================
Directional pair tests. Read means dst GPU kernel reads src GPU memory; Write means src GPU kernel writes dst GPU memory.
Src  Dst  Size        P2P        Link                Copy   BiDirCopy        Read       Write        NCCL        Host        Best  BestMethod
--------------------------------------------------------------------------------------------------------------------------------------------------
0    1    0.50 GiB    Y/Y        PXB                20.68         n/a       14.09       14.18       17.05         n/a       20.68  CUDA_COPY
0    2    0.50 GiB    Y/Y        PXB                20.68         n/a       14.10       14.18       17.03         n/a       20.68  CUDA_COPY
0    3    0.50 GiB    Y/Y        PXB                20.68         n/a       14.10       14.18       17.00         n/a       20.68  CUDA_COPY
0    4    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.43       18.96         n/a       18.96  NCCL_SENDRECV
0    5    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.43       18.85         n/a       18.85  NCCL_SENDRECV
0    6    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.43       19.04         n/a       19.04  NCCL_SENDRECV
0    7    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.43       18.96         n/a       18.96  NCCL_SENDRECV
1    0    0.50 GiB    Y/Y        CUDA_P2P           20.68         n/a       14.10       14.19       17.03         n/a       20.68  CUDA_COPY
1    2    0.50 GiB    Y/Y        PXB                20.68         n/a       14.10       14.18       17.03         n/a       20.68  CUDA_COPY
1    3    0.50 GiB    Y/Y        PXB                20.68         n/a       14.10       14.20       17.00         n/a       20.68  CUDA_COPY
1    4    0.50 GiB    Y/Y        SYS                18.56         n/a       18.57       12.45       18.97         n/a       18.97  NCCL_SENDRECV
1    5    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.46       18.96         n/a       18.96  NCCL_SENDRECV
1    6    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.44       19.12         n/a       19.12  NCCL_SENDRECV
1    7    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.46       19.06         n/a       19.06  NCCL_SENDRECV
2    0    0.50 GiB    Y/Y        CUDA_P2P           20.68         n/a       14.10       14.20       17.02         n/a       20.68  CUDA_COPY
2    1    0.50 GiB    Y/Y        PXB                20.68         n/a       14.10       14.19       17.04         n/a       20.68  CUDA_COPY
2    3    0.50 GiB    Y/Y        PIX                20.68         n/a       14.10       14.16       17.00         n/a       20.68  CUDA_COPY
2    4    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.44       19.02         n/a       19.02  NCCL_SENDRECV
2    5    0.50 GiB    Y/Y        SYS                18.56         n/a       18.54       12.43       19.07         n/a       19.07  NCCL_SENDRECV
2    6    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.44       18.96         n/a       18.96  NCCL_SENDRECV
2    7    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.44       18.93         n/a       18.93  NCCL_SENDRECV
3    0    0.50 GiB    Y/Y        CUDA_P2P           20.68         n/a       14.10       14.17       17.09         n/a       20.68  CUDA_COPY
3    1    0.50 GiB    Y/Y        PXB                20.68         n/a       14.10       14.19       17.05         n/a       20.68  CUDA_COPY
3    2    0.50 GiB    Y/Y        PIX                20.68         n/a       14.10       14.17       17.07         n/a       20.68  CUDA_COPY
3    4    0.50 GiB    Y/Y        SYS                18.56         n/a       18.57       12.50       19.03         n/a       19.03  NCCL_SENDRECV
3    5    0.50 GiB    Y/Y        SYS                18.56         n/a       18.57       12.44       18.94         n/a       18.94  NCCL_SENDRECV
3    6    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.50       19.15         n/a       19.15  NCCL_SENDRECV
3    7    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.44       18.99         n/a       18.99  NCCL_SENDRECV
4    0    0.50 GiB    Y/Y        CUDA_P2P           18.56         n/a       18.56       12.43       19.13         n/a       19.13  NCCL_SENDRECV
4    1    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.44       19.24         n/a       19.24  NCCL_SENDRECV
4    2    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.44       19.29         n/a       19.29  NCCL_SENDRECV
4    3    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.47       19.07         n/a       19.07  NCCL_SENDRECV
4    5    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.17       17.00         n/a       20.69  CUDA_COPY
4    6    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.20       16.98         n/a       20.69  CUDA_COPY
4    7    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.23       16.97         n/a       20.69  CUDA_COPY
5    0    0.50 GiB    Y/Y        CUDA_P2P           18.56         n/a       18.56       12.43       19.24         n/a       19.24  NCCL_SENDRECV
5    1    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.45       19.42         n/a       19.42  NCCL_SENDRECV
5    2    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.44       19.59         n/a       19.59  NCCL_SENDRECV
5    3    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.47       19.18         n/a       19.18  NCCL_SENDRECV
5    4    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.24       17.08         n/a       20.69  CUDA_COPY
5    6    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.25       17.08         n/a       20.69  CUDA_COPY
5    7    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.23       16.97         n/a       20.69  CUDA_COPY
6    0    0.50 GiB    Y/Y        CUDA_P2P           18.55         n/a       18.56       12.43       19.32         n/a       19.32  NCCL_SENDRECV
6    1    0.50 GiB    Y/Y        SYS                18.55         n/a       18.56       12.43       19.36         n/a       19.36  NCCL_SENDRECV
6    2    0.50 GiB    Y/Y        SYS                18.55         n/a       18.56       12.43       19.19         n/a       19.19  NCCL_SENDRECV
6    3    0.50 GiB    Y/Y        SYS                18.55         n/a       18.56       12.50       19.25         n/a       19.25  NCCL_SENDRECV
6    4    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.23       17.03         n/a       20.69  CUDA_COPY
6    5    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.20       17.03         n/a       20.69  CUDA_COPY
6    7    0.50 GiB    Y/Y        PIX                20.69         n/a       14.10       14.17       17.00         n/a       20.69  CUDA_COPY
7    0    0.50 GiB    Y/Y        CUDA_P2P           18.56         n/a       18.56       12.46       19.44         n/a       19.44  NCCL_SENDRECV
7    1    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.45       19.25         n/a       19.25  NCCL_SENDRECV
7    2    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.46       19.45         n/a       19.45  NCCL_SENDRECV
7    3    0.50 GiB    Y/Y        SYS                18.56         n/a       18.56       12.43       19.01         n/a       19.01  NCCL_SENDRECV
7    4    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.20       17.06         n/a       20.69  CUDA_COPY
7    5    0.50 GiB    Y/Y        PXB                20.69         n/a       14.10       14.23       17.01         n/a       20.69  CUDA_COPY
7    6    0.50 GiB    Y/Y        PIX                20.69         n/a       14.10       14.25       17.07         n/a       20.69  CUDA_COPY
--------------------------------------------------------------------------------------------------------------------------------------------------
P2P column is src->dst / dst->src cudaDeviceCanAccessPeer. Link comes from nvidia-smi topo -m when available.
Copy is one directed cudaMemcpyPeer payload GB/s. BiDirCopy is simultaneous src->dst plus dst->src total payload GB/s when --g2g-bidir-pair is used. Host is logical payload GB/s through pinned host memory.
BestMethod chooses the fastest measured method for that directed pair.

Best GPU-to-GPU payload bandwidth matrix, GB/s, rows are source GPUs, columns are destination GPUs:
 src\dst      GPU0      GPU1      GPU2      GPU3      GPU4      GPU5      GPU6      GPU7
    GPU0        --     20.68     20.68     20.68     18.96     18.85     19.04     18.96
    GPU1     20.68        --     20.68     20.68     18.97     18.96     19.12     19.06
    GPU2     20.68     20.68        --     20.68     19.02     19.07     18.96     18.93
    GPU3     20.68     20.68     20.68        --     19.03     18.94     19.15     18.99
    GPU4     19.13     19.24     19.29     19.07        --     20.69     20.69     20.69
    GPU5     19.24     19.42     19.59     19.18     20.69        --     20.69     20.69
    GPU6     19.32     19.36     19.19     19.25     20.69     20.69        --     20.69
    GPU7     19.44     19.25     19.45     19.01     20.69     20.69     20.69        --
Fastest directed GPU-to-GPU pair: GPU 5 -> GPU 6, 20.69 GB/s by CUDA_COPY, link PXB (PCIe, multiple PCIe switches) (CUDA cudaMemcpyPeer, driver-selected P2P path over PXB).

================ GPU-to-GPU Aggregate Summary ================
Aggregate tests launch multiple cudaMemcpyPeer transfers at the same time.
OneToAll and AllToOne are one-way aggregate payload bandwidth. BiDirAgg is send+receive payload bandwidth and is the closest number to compare with a bidirectional per-GPU NVLink spec.
GPU  Peers   Links             Slice              OneToAll       AllToOne       BiDirAgg  Notes
----------------------------------------------------------------------------------------------------------
0    7       PXBx3,SYSx4       0.50 GiB              19.42          19.68          23.05
1    7       CUDA_P2Px1,PXB... 0.50 GiB              19.42          19.68          23.57
2    7       CUDA_P2Px1,PIX... 0.50 GiB              19.42          19.68          22.82
3    7       CUDA_P2Px1,PIX... 0.50 GiB              19.42          19.68          23.98
4    7       CUDA_P2Px1,PXB... 0.50 GiB              19.42          19.68          18.69
5    7       CUDA_P2Px1,PXB... 0.50 GiB              19.41          19.68          17.95
6    7       CUDA_P2Px1,PIX... 0.50 GiB              19.42          19.68          19.12
7    7       CUDA_P2Px1,PIX... 0.50 GiB              19.42          19.68          20.13
----------------------------------------------------------------------------------------------------------
For H100 SXM/HGX, 900 GB/s is a bidirectional aggregate per-GPU number, not a single directed src->dst copy number.
NCCL version code: 23004
NCCL AllReduce buffer/rank: 0.25 GiB
NCCL AllReduce algBW: 2.05 GB/s
NCCL AllReduce busBW/link estimate: 3.58 GB/s
```
