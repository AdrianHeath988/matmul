Implements several naive and more accelerated CPU matmul implementations along with a simple kernel. Evaluations are on a ~1.95GHz throttled AMD Ryzen 7 with 16 logical cores.

- naiveMatlul works at 86.3325 million operations per second aka 0.086 GIOPS
- cachedNaiveMatMul is at 5.1 GIOPS
- cachedNaiveMatMulParallel at 14.4 GIOPS
- cachedTiledMatMulParallel at 37.2 GIOPS
- cachedTiledMatMulParallelAlligne at 43 GIOPS
- cachedTiledMatMulAVX at 74 GIOPS
- cachedTiledMatMulAVX_Ultimate at 118 GIOPS (~95% of theoretical max)
- simple naive matmul kernel does 370 GIOPs on GeForce RTX 5060 (Blackwell architecture)
If you have suggestions or other benchmarks, feel free to add :)
