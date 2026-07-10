package gemm_test

import (
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/gemm/gemm"
)

func matmulDarwin(b *testing.B, rows, cols int) {
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

func BenchmarkMatMulDarwin1024(b *testing.B) {
	matmulDarwin(b, 1024, 1024)
}

func BenchmarkMatMulDarwin2048(b *testing.B) {
	matmulDarwin(b, 2048, 2048)
}

func BenchmarkMatMulDarwin4096(b *testing.B) {
	matmulDarwin(b, 4096, 4096)
}
