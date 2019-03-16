package bintree

import (
	"context"
	"flag"
	"fmt"
	"math"
	"strconv"
	"testing"
	"time"
)

var (
	NNodes  = flag.Int("nodes", 1000, "number of internal nodes.")
	MaxTime = flag.Duration("maxtime", 1000*time.Millisecond, "context timeout (in ms.) for evop.")
	MaxIter = flag.Int("maxiter", 1000, "maximum number of iterations with no progress.")
	Debug   = flag.Bool("debug", false, "debug output")

	InvertionRate  = flag.Float64("invertion", 0.2, "invertion rate")
	SwapRate       = flag.Float64("swap", 0.2, "swap rate")
	CrossoverRate  = flag.Float64("crossover", 0.2, "crossover rate")
	SplayLeftRate  = flag.Float64("splayleft", 0.2, "splay left rate")
	SplayRightRate = flag.Float64("splayright", 02, "splay right rate")
)

func init() {
	flag.Parse()
	fmt.Println("BinTree:")
	fmt.Printf("\tnodes: %d (total: %d)\n", *NNodes, 2*(*NNodes)+1)
	fmt.Printf("\tmaxtime: %v\n", *MaxTime)
	fmt.Printf("\tmaxiter: %d\n", *MaxIter)
	fmt.Printf("\tdebug: %v\n", *Debug)
	fmt.Println("Rates:")
	fmt.Printf("\tinvertion: %.2f\n", *InvertionRate)
	fmt.Printf("\tswap: %.2f\n", *SwapRate)
	fmt.Printf("\tcrossover: %.2f\n", *CrossoverRate)
	fmt.Printf("\tsplay left: %.2f\n", *SplayLeftRate)
	fmt.Printf("\tsplay right: %.2f\n", *SplayRightRate)
}

func TestOptimizeListBST(t *testing.T) {
	for nomiss := false; ; nomiss = !nomiss {
		bst, genome := randListBST(*NNodes, nomiss)
		val := eval(genome)
		t.Logf("eval: %d\n", val)
		if *Debug {
			fmt.Println("BST:", bst)
		}

		ctx, cancel := context.WithTimeout(context.Background(), *MaxTime)
		defer cancel()
		id := newIndividual(genome).
			withOperator(inversion, float32(*InvertionRate)).
			withOperator(swap, float32(*SwapRate)).
			withOperator(crossover, float32(*CrossoverRate)).
			withOperator(splayLeft, float32(*SplayLeftRate)).
			withOperator(splayRight, float32(*SplayRightRate)).
			withConstraint(isBinTree).
			withConstraint(isBST)

		t.Run("evop.optimize", func(*testing.T) {
			t0 := time.Now()
			min, err := id.optimize(ctx, *MaxIter)
			t.Logf("[%v] evop.optimize (nomiss: %v): %d %v\n", time.Now().Sub(t0), nomiss, min, err)
			if *Debug {
				fmt.Println(newBinTree(id.genome))
			}
		})

		t.Run("bintree.optimize", func(*testing.T) {
			t0 := time.Now()
			min := bst.optimize()
			t.Logf("[%v] binTree.optimize (nomiss: %v): %d\n", time.Now().Sub(t0), nomiss, min)
			if *Debug {
				fmt.Println(bst)
			}
		})

		if nomiss {
			break
		}
	}
}

func TestOptimizeNonOptimalBST(t *testing.T) {
	for nomiss := false; ; nomiss = !nomiss {
		bst, genome := randNonOptimalBST(*NNodes, nomiss)
		val := eval(genome)
		t.Logf("eval: %d\n", val)
		if *Debug {
			fmt.Println("BST:", bst)
		}

		ctx, cancel := context.WithTimeout(context.Background(), *MaxTime)
		defer cancel()
		id := newIndividual(genome).
			withOperator(inversion, float32(*InvertionRate)).
			withOperator(swap, float32(*SwapRate)).
			withOperator(crossover, float32(*CrossoverRate)).
			withOperator(splayLeft, float32(*SplayLeftRate)).
			withOperator(splayRight, float32(*SplayRightRate)).
			withConstraint(isBinTree).
			withConstraint(isBST)

		t.Run("evop.optimize", func(*testing.T) {
			t0 := time.Now()
			min, err := id.optimize(ctx, *MaxIter)
			t.Logf("[%v] evop.optimize (nomiss: %v): %d %v\n", time.Now().Sub(t0), nomiss, min, err)
			if *Debug {
				fmt.Println(newBinTree(id.genome))
			}
		})

		t.Run("bintree.optimize", func(*testing.T) {
			t0 := time.Now()
			min := bst.optimize()
			t.Logf("[%v] binTree.optimize (nomiss: %v): %d\n", time.Now().Sub(t0), nomiss, min)
			if *Debug {
				fmt.Println(bst)
			}
		})

		if nomiss {
			break
		}
	}
}

func TestOptimizeBalancedBST(t *testing.T) {
	for nomiss := false; ; nomiss = !nomiss {
		bst, genome := randBalancedBST(*NNodes, nomiss)
		val := eval(genome)
		t.Logf("eval: %d\n", val)
		if *Debug {
			fmt.Println("BST:", bst)
		}

		ctx, cancel := context.WithTimeout(context.Background(), *MaxTime)
		defer cancel()
		id := newIndividual(genome).
			withOperator(inversion, float32(*InvertionRate)).
			withOperator(swap, float32(*SwapRate)).
			withOperator(crossover, float32(*CrossoverRate)).
			withOperator(splayLeft, float32(*SplayLeftRate)).
			withOperator(splayRight, float32(*SplayRightRate)).
			withConstraint(isBinTree).
			withConstraint(isBST)

		t.Run("evop.optimize", func(*testing.T) {
			t0 := time.Now()
			min, err := id.optimize(ctx, *MaxIter)
			t.Logf("[%v] evop.optimize (nomiss: %v): %d %v\n", time.Now().Sub(t0), nomiss, min, err)
			if *Debug {
				fmt.Println(newBinTree(id.genome))
			}
		})

		t.Run("bintree.optimize", func(*testing.T) {
			t0 := time.Now()
			min := bst.optimize()
			t.Logf("[%v] binTree.optimize (nomiss: %v): %d\n", time.Now().Sub(t0), nomiss, min)
			if *Debug {
				fmt.Println(bst)
			}
		})

		if nomiss {
			break
		}
	}
}

func TestSplayRight(t *testing.T) {
	tests := []struct {
		g        string
		expected string
	}{
		{
			g:        "111100000",
			expected: "111000100",
		},
		{
			g:        "111000100",
			expected: "110010100",
		},
		{
			g:        "110010100",
			expected: "101010100",
		},
		{
			g:        "110100100",
			expected: "101100100",
		},
		{
			g:        "100",
			expected: "100",
		},
		{
			g:        "111001010010100",
			expected: "110011010010100",
		},
	}

	for i, ti := range tests {
		g := splayRight(genome(t.Helper, ti.g))

		actual, expected := fmt.Sprintf("%s", g), fmt.Sprintf("%s", genome(t.Helper, ti.expected))
		if expected != actual {
			t.Errorf("[%d] splayRight(%s):\n\tactual: %s\n\texpected: %s\n", i, g, actual, expected)
		}
	}
}

func TestSelectOperator(t *testing.T) {
	stat := make(map[string]int)

	id := newIndividual(nil).
		withOperator(func(g []*gene) []*gene {
			stat["I"]++
			return g
		}, float32(*InvertionRate)).
		withOperator(func(g []*gene) []*gene {
			stat["S"]++
			return g
		}, float32(*SwapRate)).
		withOperator(func(g []*gene) []*gene {
			stat["X"]++
			return g
		}, float32(*CrossoverRate)).
		withOperator(func(g []*gene) []*gene {
			stat["SL"]++
			return g
		}, float32(*SplayLeftRate)).
		withOperator(func(g []*gene) []*gene {
			stat["SR"]++
			return g
		}, float32(*SplayRightRate))

	for i := 0; i < 100; i++ {
		id.selectOperator()(nil)
	}

	for o, c := range stat {
		t.Logf("%s: %d\n", o, c)
	}
}

func TestSplayLeft(t *testing.T) {
	tests := []struct {
		g        string
		expected string
	}{
		{
			g:        "1011000",
			expected: "1101000",
		},
		{
			g:        "111000100",
			expected: "111100000",
		},
		{
			g:        "111100000",
			expected: "111100000",
		},
		{
			g:        "111001010010100",
			expected: "111100101000100",
		},
		{
			g:        "1010100",
			expected: "1100100",
		},
		{
			g:        "1100100",
			expected: "1110000",
		},
		{
			g:        "100",
			expected: "100",
		},
		{
			g:        "111111000000100",
			expected: "111111100000000",
		},
	}

	for i, ti := range tests {
		g := splayLeft(genome(t.Helper, ti.g))

		actual, expected := fmt.Sprintf("%s", g), fmt.Sprintf("%s", genome(t.Helper, ti.expected))
		if expected != actual {
			t.Errorf("[%d] splayLeft(%s):\n\tactual: %s\n\texpected: %s\n", i, g, actual, expected)
		}
	}
}

func TestCrossoverAt(t *testing.T) {
	tests := []struct {
		g        string
		expected string
		k        int
	}{
		{
			g:        "10110011000",
			expected: "01100110100",
			k:        0,
		},
		{
			g:        "10110011000",
			expected: "11001101000",
			k:        1,
		},
		{
			g:        "10110011000",
			expected: "10011010100",
			k:        2,
		},
		{
			g:        "10110011000",
			expected: "00110101100",
			k:        3,
		},
		{
			g:        "10110011000",
			expected: "01101011000",
			k:        4,
		},
		{
			g:        "10110011000",
			expected: "11010110000",
			k:        5,
		},
		{
			g:        "10110011000",
			expected: "10101100100",
			k:        6,
		},
		{
			g:        "10110011000",
			expected: "01011001100",
			k:        7,
		},
	}

	for i, ti := range tests {
		g := crossoverAt(genome(t.Helper, ti.g), ti.k)

		act, exp := fmt.Sprintf("%s", g), fmt.Sprintf("%s", genome(t.Helper, ti.expected))
		if exp != act {
			t.Errorf("[%d] crossoverAt(%s, %d):\n\tactual: %s\n\texpected: %s\n", i, ti.g, ti.k, act, ti.expected)
		}
	}
}

func TestSwapAt(t *testing.T) {
	tests := []struct {
		g        string
		expected string
		k1       int
		k2       int
	}{
		{
			g:        "11110000",
			expected: "10110001",
			k1:       1,
			k2:       7,
		},
		{
			g:        "11110000",
			expected: "11010010",
			k1:       2,
			k2:       6,
		},
		{
			g:        "11110000",
			expected: "11100100",
			k1:       3,
			k2:       5,
		},
		{
			g:        "11110000",
			expected: "01110100",
			k1:       0,
			k2:       5,
		},
		{
			g:        "11110000",
			expected: "11100001",
			k1:       3,
			k2:       7,
		},
		{
			g:        "11110000",
			expected: "01110001",
			k1:       0,
			k2:       7,
		},
		{
			g:        "11110000",
			expected: "11110000",
			k1:       7,
			k2:       5,
		},
	}

	for i, ti := range tests {
		g := swapAt(genome(t.Helper, ti.g), ti.k1, ti.k2)

		act, exp := fmt.Sprintf("%s", g), fmt.Sprintf("%s", genome(t.Helper, ti.expected))
		if exp != act {
			t.Errorf("[%d] swapAt(%s, %d, %d):\n\tactual: %s\n\texpected: %s\n", i, ti.g, ti.k1, ti.k2, act, exp)
		}
	}
}

func TestInvertionAt(t *testing.T) {
	tests := []struct {
		g        string
		expected string
		k1       int
		k2       int
	}{
		{
			g:        "11110000",
			expected: "10000111",
			k1:       1,
			k2:       7,
		},
		{
			g:        "11110000",
			expected: "11000110",
			k1:       2,
			k2:       6,
		},
		{
			g:        "11110000",
			expected: "11100100",
			k1:       3,
			k2:       5,
		},
		{
			g:        "11110000",
			expected: "00111100",
			k1:       0,
			k2:       5,
		},
		{
			g:        "11110000",
			expected: "11100001",
			k1:       3,
			k2:       7,
		},
		{
			g:        "11110000",
			expected: "00001111",
			k1:       0,
			k2:       7,
		},
		{
			g:        "11110000",
			expected: "11110000",
			k1:       5,
			k2:       7,
		},
	}

	for i, ti := range tests {
		g := inversionAt(genome(t.Helper, ti.g), ti.k1, ti.k2)

		act, exp := fmt.Sprintf("%s", g), fmt.Sprintf("%s", genome(t.Helper, ti.expected))
		if exp != act {
			t.Errorf("[%d] invertionAt(%s, %d, %d):\n\tactual: %s\n\texpected: %s\n", i, ti.g, ti.k1, ti.k2, act, exp)
		}
	}
}

func TestNewBinTree(t *testing.T) {
	tests := []struct {
		g  string
		n  int
		ln int
		rn int
	}{
		{
			g:  "111100000",
			n:  9,
			ln: 7,
			rn: 1,
		},
		{
			g:  "111000100",
			n:  9,
			ln: 5,
			rn: 3,
		},
		{
			g:  "110010100",
			n:  9,
			ln: 3,
			rn: 5,
		},
		{
			g:  "100",
			n:  3,
			ln: 1,
			rn: 1,
		},
		{
			g:  "0",
			n:  1,
			ln: 0,
			rn: 0,
		},
	}

	for i, ti := range tests {
		bt := newBinTree(genome(t.Helper, ti.g))

		if ti.n != bt.n {
			t.Errorf("[%d] newBinTree(%s):\n\tactual: %d\n\texpected: %d\n", i, ti.g, bt.n, ti.n)
		}

		ln := 0
		if bt.left != nil {
			ln = bt.left.n
		}
		if ti.ln != ln {
			t.Errorf("[%d] newBinTree(%s).left:\n\tactual: %d\n\texpected: %d\n", i, ti.g, ln, ti.ln)
		}

		rn := 0
		if bt.right != nil {
			rn = bt.right.n
		}
		if ti.rn != rn {
			t.Errorf("[%d] newBinTree(%s).right:\n\tactual: %d\n\texpected: %d\n", i, ti.g, rn, ti.rn)
		}
	}
}

func TestIsBinTree(t *testing.T) {
	tests := []struct {
		g        string
		expected bool
	}{
		{
			g:        "111100000",
			expected: true,
		},
		{
			g:        "111000100",
			expected: true,
		},
		{
			g:        "110010100",
			expected: true,
		},
		{
			g:        "110100100",
			expected: true,
		},
		{
			g:        "100",
			expected: true,
		},
		{
			g:        "111001010010100",
			expected: true,
		},
		{
			g:        "100001010010100",
			expected: false,
		},
		{
			g:        "001",
			expected: false,
		},
	}

	for i, ti := range tests {
		actual := isBinTree(genome(t.Helper, ti.g))

		if ti.expected != actual {
			t.Errorf("[%d] isBinTree(%s):\n\tactual: %v\n\texpected: %v\n", i, ti.g, actual, ti.expected)
		}
	}
}

func TestIsBST(t *testing.T) {
	tests := []struct {
		g        []*gene
		expected bool
	}{
		{
			g:        genome(t.Helper, "abcd00000"),
			expected: false,
		},
		{
			g:        genome(t.Helper, "dcb000h00"),
			expected: true,
		},
		{
			g:        genome(t.Helper, "ba00c0d00"),
			expected: true,
		},
		{
			g:        genome(t.Helper, "ba0c00d00"),
			expected: false,
		},
		{
			g:        genome(t.Helper, "a00"),
			expected: true,
		},
		{
			g:        genome(t.Helper, "111001010010100"),
			expected: true,
		},
		{
			g:        genome(t.Helper, "mfda00cb00g00hd00i00n00"),
			expected: false,
		},
	}

	for i, ti := range tests {
		actual := isBST(ti.g)

		if ti.expected != actual {
			t.Errorf("[%d] isBST(%s):\n\tactual: %v\n\texpected: %v\n", i, ti.g, actual, ti.expected)
		}
	}
}

func TestEval(t *testing.T) {
	tests := []struct {
		g        []*gene
		expected int
	}{
		{
			g:        genome(t.Helper, "111100000"),
			expected: 10,
		},
		{
			g:        genome(t.Helper, "111000100"),
			expected: 8,
		},
		{
			g:        genome(t.Helper, "110010100"),
			expected: 8,
		},
		{
			g:        genome(t.Helper, "110100100"),
			expected: 8,
		},
		{
			g:        genome(t.Helper, "100"),
			expected: 1,
		},
		{
			g:        genome(t.Helper, "111001010010100"),
			expected: 18,
		},
		{
			g: []*gene{
				&gene{key: "a", weight: 4},
				&gene{key: "b", weight: 5},
				&gene{key: "c", weight: 6},
				&gene{key: "d", weight: 7},
				&gene{key: "e", weight: 8},
				&gene{key: "f", weight: 9},
				&gene{key: "g", weight: 10},
				nil, nil, nil, &gene{weight: 13}, nil, nil, nil, nil},
			expected: 302,
		},
	}

	for i, tc := range tests {
		actual := eval(tc.g)
		if tc.expected != actual {
			t.Errorf("[%d] eval(%s):\n\texpected: %d\n\tactual: %d\n", i, tc.g, tc.expected, actual)
		}
	}
}

func TestBinTreeOptimize(t *testing.T) {
	var genome = []*gene{
		&gene{
			key:    "k2",
			weight: 10,
		},
		&gene{
			key:    "k1",
			weight: 15,
		}, &gene{
			weight: 5,
		}, &gene{
			weight: 10,
		},
		&gene{
			key:    "k4",
			weight: 10,
		},
		&gene{
			key:    "k3",
			weight: 5,
		}, &gene{
			weight: 5,
		}, &gene{
			weight: 5,
		},
		&gene{
			key:    "k5",
			weight: 20,
		}, &gene{
			weight: 5,
		}, &gene{
			weight: 10,
		},
	}

	n := 11
	bt := newBinTree(genome)
	if n != bt.n {
		t.Errorf("binTree(%s)\n%s\nn:\n\tactual: %d\n\texpected: %d\n", genome, bt.String(), bt.n, n)
	}

	min := 275
	if val := bt.optimize(); min != val {
		t.Errorf("binTree(%s)\n%s\noptimize:\n\tactual: %d\n\texpected: %d\n", genome, bt.String(), val, min)
	}
	if n != bt.n {
		t.Errorf("binTree(%s)\n%s\nn:\n\tactual: %d\n\texpected: %d\n", genome, bt.String(), bt.n, n)
	}
}

func TestListBST(t *testing.T) {
	expected := 2*(*NNodes) + 1
	bt, genome := randListBST(*NNodes, false)
	if bt.n != expected {
		t.Errorf("binTree(%s)\n%s\noptimize:\n\tactual: %d\n\texpected: %d\n", genome, bt.String(), bt.n, expected)
	}
	if *Debug {
		fmt.Println(bt)
	}
}

func TestNonOptimalBST(t *testing.T) {
	expected := 2*(*NNodes) + 1
	bt, genome := randNonOptimalBST(*NNodes, false)
	if bt.n != expected {
		t.Errorf("binTree(%s)\n%s\noptimize:\n\tactual: %d\n\texpected: %d\n", genome, bt.String(), bt.n, expected)
	}
	if *Debug {
		fmt.Println(bt)
	}
}

func TestBalancedBST(t *testing.T) {
	expected := 2*(*NNodes) + 1
	bt, genome := randBalancedBST(*NNodes, false)
	if bt.n != expected {
		t.Errorf("binTree(%s)\n%s\noptimize:\n\tactual: %d\n\texpected: %d\n", genome, bt.String(), bt.n, expected)
	}
	if *Debug {
		fmt.Println(bt)
	}
}

func genome(helper func(), s string) (g []*gene) {
	helper()

	for _, c := range s {
		if c == ' ' {
			continue
		}

		if c == '0' {
			g = append(g, nil)
		} else {
			g = append(g, &gene{
				key:    string(c),
				weight: 1,
			})
		}
	}
	return g
}

// randListBST generates a binary search tree where n internal nodes are linked like in a list.
// The function is used only for testing.
/*
(k4/1)
   |-(0)
    -(k3/3)
         |-(0)
          -(k2/3)
               |-(0)
                -(k1/2)
                     |-(0)
                      -(1)
*/
func randListBST(n int, nomiss bool) (*binTree, []*gene) {
	w := len(strconv.Itoa(n))
	genome := make([]*gene, 2*n+1)
	for i := 0; i < n; i++ {
		genome[i] = &gene{
			key:    fmt.Sprintf("k%0*d", w, n-i),
			weight: randIntn(n),
		}
	}

	if !nomiss {
		q := int(math.Log2(float64(n)))
		for i := n; i < 2*n+1; i++ {
			genome[i] = &gene{
				weight: randIntn(q),
			}
		}
	}

	return newBinTree(genome), genome
}

// randNonOptimalBST generates a non optimal binary search tree.
// A BST is like a list with the highest weights at the bottom
// The function is used only for testing.
/*
(k5/0)
   |-(0)
    -(k4/1)
         |-(0)
          -(k3/2)
               |-(1)
                -(k2/3)
                     |-(1)
                      -(k1/4)
                           |-(0)
                            -(0)
*/
func randNonOptimalBST(n int, nomiss bool) (*binTree, []*gene) {
	w := len(strconv.Itoa(n))
	genome := make([]*gene, 2*n+1)
	for i := 0; i < n; i++ {
		genome[i] = &gene{
			key:    fmt.Sprintf("k%0*d", w, n-i),
			weight: i,
		}
	}

	if !nomiss {
		q := int(math.Log2(float64(n)))
		for i := n; i < 2*n+1; i++ {
			genome[i] = &gene{
				weight: randIntn(q),
			}
		}
	}

	return newBinTree(genome), genome
}

// randBalancedBST generates a balanced binary search tree with n internal nodes.
// The function is used only for testing.
/*
(k3/3)
   |-(k4/0)
   |     |-(k5/2)
   |     |     |-(1)
   |     |      -(0)
   |      -(0)
    -(k1/4)
         |-(k2/0)
         |     |-(1)
         |      -(0)
          -(0)
*/
func randBalancedBST(n int, nomiss bool) (*binTree, []*gene) {
	q := int(math.Log2(float64(n)))
	w := len(strconv.Itoa(n))
	genome := make([]*gene, 2*n+1)

	var (
		off   int
		build func(l, r int)
	)
	build = func(l, r int) {
		if l > r || r < l {
			genome[off] = &gene{weight: randIntn(q)}
			return
		}

		mid := (l + r) / 2
		genome[off] = &gene{
			key:    fmt.Sprintf("k%0*d", w, mid),
			weight: randIntn(n),
		}

		off++
		build(l, mid-1)

		off++
		build(mid+1, r)
	}
	build(1, n)

	return newBinTree(genome), genome
}
