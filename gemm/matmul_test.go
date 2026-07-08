package gemm_test

import (
	"testing"

	"github.com/itsubaki/autograd/tensor"
)

func matmul(b *testing.B, rows, cols int) {
	data := make([]float64, rows*cols)
	t := tensor.New([]int{rows, cols}, data)

	b.ReportAllocs()
	b.ResetTimer()

	var out *tensor.Tensor[float64]
	for b.Loop() {
		out = tensor.MatMul(t, t)
	}

	_ = out
}

func BenchmarkMatMul1024(b *testing.B) {
	matmul(b, 1024, 1024)
}

func BenchmarkMatMul2048(b *testing.B) {
	matmul(b, 2048, 2048)
}

func BenchmarkMatMul4096(b *testing.B) {
	matmul(b, 4096, 4096)
}
