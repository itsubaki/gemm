package gemm_test

import (
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/gemm/gemm"
)

func matmulDarwinARM64(b *testing.B, rows, cols int) {
	data := make([]float64, rows*cols)
	t := tensor.New([]int{rows, cols}, data)

	b.ReportAllocs()
	b.ResetTimer()

	var out *tensor.Tensor[float64]
	for b.Loop() {
		out = gemm.MatMul(t, t)
	}

	_ = out
}

func BenchmarkMatMulDarwinARM641024(b *testing.B) {
	matmulDarwinARM64(b, 1024, 1024)
}

func BenchmarkMatMulDarwinARM642048(b *testing.B) {
	matmulDarwinARM64(b, 2048, 2048)
}

func BenchmarkMatMulDarwinARM644096(b *testing.B) {
	matmulDarwinARM64(b, 4096, 4096)
}
