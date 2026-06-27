package main

/*
#cgo CFLAGS: -O3 -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/itsubaki/autograd/tensor"
)

func matmul(v, w *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	m, n, k := v.Shape[0], w.Shape[1], v.Shape[1]
	c := make([]float64, m*n)
	alpha := C.double(1.0)
	beta := C.double(0.0)

	C.cblas_dgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k),
		alpha,
		(*C.double)(unsafe.Pointer(&v.Data[0])), C.int(m),
		(*C.double)(unsafe.Pointer(&w.Data[0])), C.int(k),
		beta,
		(*C.double)(unsafe.Pointer(&c[0])), C.int(m),
	)

	return tensor.New([]int{m, n}, c)
}

func main() {
	v := tensor.New([]int{2, 2}, []float64{
		1, 2,
		3, 4,
	})

	w := tensor.New([]int{2, 2}, []float64{
		5, 6,
		7, 8,
	})

	result := matmul(v, w)

	// 19 22
	// 43 50
	fmt.Println(result)
}
