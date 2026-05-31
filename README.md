./gpu_stress_nccl --help
Usage: ./gpu_stress_nccl [options]
Options:
  --single-seconds <sec> Seconds for each single-GPU memory stress test. Default: 30
  --single-gpu-seconds <sec> Alias for --single-seconds
  --all-seconds <sec>    Seconds for concurrent all-GPU memory stress and NCCL link test. Default: same as --single-seconds
  --all-gpu-seconds <sec> Alias for --all-seconds
  --concurrent-seconds <sec> Alias for --all-seconds
  --seconds <sec>        Backward-compatible alias for --single-seconds
  --reserve-mb <MB>     Keep this much free memory per GPU for CUDA/NCCL/OS. Default: 512
  --nccl-mb <MB>        NCCL send buffer size per GPU. It uses the same size for recv. Default: 256
  --nccl-batch <N>      Queue N AllReduce ops before synchronizing. Default: 4
  --temp-threshold <C> Warn when any GPU temperature reaches/exceeds this value. Default: 80
  --temp-interval-ms <N> Poll GPU temperature every N milliseconds. Default: 1000
  --no-temp-monitor    Disable GPU temperature monitoring
  --g2g-seconds <sec>  Seconds for each method in each directed GPU-to-GPU pair. Default: min(1, all-GPU seconds)
  --peer-seconds <sec> Alias for --g2g-seconds
  --g2g-mb <MB>        Buffer size for each GPU-to-GPU pair test. Default: 512
  --peer-mb <MB>       Alias for --g2g-mb
  --g2g-batch <N>      Queue N pair transfers/kernels before synchronizing. Default: 4
  --g2g-host-always    Also benchmark pinned host-staged path even when CUDA P2P is available
  --g2g-bidir-pair     Also measure simultaneous bidirectional cudaMemcpyPeer for each GPU pair
  --no-g2g-pair-test   Disable directed pair matrix tests, but keep aggregate tests if enabled
  --g2g-aggregate-seconds <sec> Seconds for per-GPU aggregate one-to-all/all-to-one/bidir tests. Default: min(1, g2g seconds)
  --no-g2g-aggregate-test Disable per-GPU aggregate GPU-to-GPU tests
  --no-g2g-test        Disable pairwise and aggregate GPU-to-GPU bandwidth tests
  --help                Show this help
