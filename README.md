gpu_stress_nccl usage:
<img width="1028" height="390" alt="image" src="https://github.com/user-attachments/assets/cf43dbc1-22f2-4f29-803f-fb45e60e7500" />
<br>
nvcc -O3 -std=c++17 gpu_stress_nccl_g2g_link.cu -o gpu_stress_nccl -ldl -Xcompiler -pthread
<br>
H100/800
<br>
nvcc -O3 -std=c++17   -gencode=arch=compute_90,code=sm_90   gpu_stress_nccl_g2g_link.cu   -o gpu_stress_nccl   -ldl -Xcompiler -pthread
