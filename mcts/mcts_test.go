package mcts

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
	BST     = flag.Bool("bst", false, "binary tree satisfies bst constraint.")
	Debug   = flag.Bool("debug", false, "debug output.")
)

func init() {
	flag.Parse()
	fmt.Println("mcts:")
	fmt.Printf("\tnodes: %d (total: %d)\n", *NNodes, 2*(*NNodes)+1)
	fmt.Printf("\tmaxtime: %v\n", *MaxTime)
	fmt.Printf("\tmaxiter: %d\n", *MaxIter)
	fmt.Printf("\tbst: %v\n", *BST)
	fmt.Printf("\tdebug: %v\n", *Debug)
}

func TestOptimizeListBST(t *testing.T) {
	for _, nomiss := range []bool{false, true} {
		bst, genome := randListBST(*NNodes, nomiss)
		val := eval(genome)
		t.Logf("eval: %d\n", val)
		if *Debug {
			fmt.Println(bst)
		}

		ctx, cancel := context.WithTimeout(context.Background(), *MaxTime)
		defer cancel()
		mcts := newMCTS(genome).
			withOperator(crossover).
			withOperator(inversion).
			withOperator(swap).
			withOperator(splayLeft).
			withOperator(splayRight).
			withConstraint(isBinTree)
		if *BST {
			mcts = mcts.withConstraint(isBST)
		}

		t.Run("evop.optimize", func(*testing.T) {
			t0 := time.Now()
			min, err := mcts.optimize(ctx, *MaxIter)
			t.Logf("[%v] evop.optimize (nomiss: %v): %d %v\n", time.Now().Sub(t0), nomiss, min, err)
			if *Debug {
				fmt.Println(newBinTree(mcts.s0.genome))
				fmt.Println(mcts.String())
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
	}
}

func TestOptimizeNonOptimalBST(t *testing.T) {
	for _, nomiss := range []bool{false, true} {
		bst, genome := randNonOptimalBST(*NNodes, nomiss)
		val := eval(genome)
		t.Logf("eval: %d\n", val)
		if *Debug {
			fmt.Println("BST:", bst)
		}

		ctx, cancel := context.WithTimeout(context.Background(), *MaxTime)
		defer cancel()
		mcts := newMCTS(genome).
			withOperator(crossover).
			withOperator(inversion).
			withOperator(swap).
			withOperator(splayLeft).
			withOperator(splayRight).
			withConstraint(isBinTree)
		if *BST {
			mcts = mcts.withConstraint(isBST)
		}

		t.Run("evop.optimize", func(*testing.T) {
			t0 := time.Now()
			min, err := mcts.optimize(ctx, *MaxIter)
			t.Logf("[%v] evop.optimize (nomiss: %v): %d %v\n", time.Now().Sub(t0), nomiss, min, err)
			if *Debug {
				fmt.Println(newBinTree(mcts.s0.genome))
				fmt.Println(mcts.String())
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
	}
}

func TestOptimizeBalancedBST(t *testing.T) {
	for _, nomiss := range []bool{false, true} {
		bst, genome := randBalancedBST(*NNodes, nomiss)
		val := eval(genome)
		t.Logf("eval: %d\n", val)
		if *Debug {
			fmt.Println("BST:", bst)
		}

		ctx, cancel := context.WithTimeout(context.Background(), *MaxTime)
		defer cancel()
		mcts := newMCTS(genome).
			withOperator(crossover).
			withOperator(inversion).
			withOperator(swap).
			withOperator(splayLeft).
			withOperator(splayRight).
			withConstraint(isBinTree)
		if *BST {
			mcts = mcts.withConstraint(isBST)
		}

		t.Run("evop.optimize", func(*testing.T) {
			t0 := time.Now()
			min, err := mcts.optimize(ctx, *MaxIter)
			t.Logf("[%v] evop.optimize (nomiss: %v): %d %v\n", time.Now().Sub(t0), nomiss, min, err)
			if *Debug {
				fmt.Println(newBinTree(mcts.s0.genome))
				fmt.Println(mcts.String())
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
	}
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
