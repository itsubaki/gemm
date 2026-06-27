package main

import (
	"fmt"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/gemm/gemm"
)

func main() {
	v := tensor.New([]int{2, 2}, []float64{
		1, 2,
		3, 4,
	})

	w := tensor.New([]int{2, 2}, []float64{
		5, 6,
		7, 8,
	})

	result := gemm.MatMul(v, w)

	// 19 22
	// 43 50
	fmt.Println(result)
}
