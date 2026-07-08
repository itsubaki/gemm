//go:build !(darwin && arm64)

package gemm

import "github.com/itsubaki/autograd/tensor"

func MatMul(v, w *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	return tensor.MatMul(v, w)
}
