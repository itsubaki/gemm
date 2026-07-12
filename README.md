# gemm

```shell
% make bench
go test -bench . ./... --benchmem
?       github.com/itsubaki/gemm        [no test files]
goos: darwin
goarch: arm64
pkg: github.com/itsubaki/gemm/gemm
cpu: Apple M3
BenchmarkMatMulDarwin1024-8     202           5856426 ns/op         8388810 B/op         4 allocs/op
BenchmarkMatMulDarwin2048-8      22          52009826 ns/op        33554554 B/op         4 allocs/op
BenchmarkMatMulDarwin4096-8       3         435281319 ns/op       134217840 B/op         4 allocs/op
BenchmarkMatMul1024-8             1        1161112709 ns/op         8390080 B/op        22 allocs/op
BenchmarkMatMul2048-8             1        8385113667 ns/op        33555552 B/op        21 allocs/op
BenchmarkMatMul4096-8             1       66946933417 ns/op       134218336 B/op        18 allocs/op
```
