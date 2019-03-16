package bintree

import (
	"math/rand"
	"time"
)

// random contains source seed based on timestamp.
var random = rand.New(rand.NewSource(time.Now().UnixNano()))

// randFloat32 generates an float32 in [0, 1) with uniform distribution.
func randFloat32() float32 {
	return random.Float32()
}

// randIntn generates an int in [0, n) with uniform distribution.
func randIntn(n int) int {
	return random.Intn(n)
}

// randNormIntn generates an int in [0, n)
// with normal distribution (mean = 0.5 * n, stddev = 0.15 * n)
func randNormIntn(n int) int {
	f := float64(n)
	mean, stddev := 0.5*f, 0.15*f
	for {
		if g := int(random.NormFloat64()*stddev + mean); g > 0 {
			if g >= n {
				g = g % n
			}
			return g
		}
	}
}
