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

func main() {
	m, n, k := 2, 2, 2

	a := []C.double{
		1.0, 2.0,
		3.0, 4.0,
	}
	b := []C.double{
		5.0, 6.0,
		7.0, 8.0,
	}

	c := make([]C.double, m*n)
	alpha := C.double(1.0)
	beta := C.double(0.0)

	C.cblas_dgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k),
		alpha,
		(*C.double)(unsafe.Pointer(&a[0])), C.int(m),
		(*C.double)(unsafe.Pointer(&b[0])), C.int(k),
		beta,
		(*C.double)(unsafe.Pointer(&c[0])), C.int(m),
	)

	fmt.Println(c[0], c[1])
	fmt.Println(c[2], c[3])
}
