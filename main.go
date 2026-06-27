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
)

type Tensor struct {
	Data  []float64
	Shape []int
}

func matmul(v, w Tensor) Tensor {
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

	return Tensor{
		Data:  c,
		Shape: []int{m, n},
	}
}

func main() {
	v := Tensor{
		Data: []float64{
			1, 2,
			3, 4,
		},
		Shape: []int{2, 2},
	}

	w := Tensor{
		Data: []float64{
			5, 6,
			7, 8,
		},
		Shape: []int{2, 2},
	}

	result := matmul(v, w)

	// 19 22
	// 43 50
	fmt.Println(result)
}
