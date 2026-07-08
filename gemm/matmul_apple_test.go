package gemm_test

import (
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/gemm/gemm"
)

func matmulApple(b *testing.B, rows, cols int) {
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

func BenchmarkMatMulApple1024(b *testing.B) {
	matmulApple(b, 1024, 1024)
}

func BenchmarkMatMulApple2048(b *testing.B) {
	matmulApple(b, 2048, 2048)
}

func BenchmarkMatMulApple4096(b *testing.B) {
	matmulApple(b, 4096, 4096)
}
