//go:build darwin

package gemm

/*
#cgo CFLAGS: -O3 -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import (
	"unsafe"

	"github.com/itsubaki/autograd/tensor"
)

func MatMul(v, w *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	if len(v.Shape) != 2 || len(w.Shape) != 2 {
		panic("invalid shape")
	}

	if v.Shape[1] != w.Shape[0] {
		panic("shape mismatch")
	}

	m, k, n := v.Shape[0], v.Shape[1], w.Shape[1]
	alpha := C.double(1.0)
	beta := C.double(0.0)

	out := make([]float64, m*n)
	C.cblas_dgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		C.int(m),
		C.int(n),
		C.int(k),
		alpha,
		(*C.double)(unsafe.Pointer(&v.Data[0])),
		C.int(k),
		(*C.double)(unsafe.Pointer(&w.Data[0])),
		C.int(n),
		beta,
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	return tensor.New([]int{m, n}, out)
}
