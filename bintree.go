package evop

import (
	"context"
	"fmt"
	"math"
	"strings"
)

type (
	individual struct {
		genome []*gene
		ops    []operator
		rates  []float32
		consts []constraint
	}
)

// newIndividual creates a new individual from the genome.
func newIndividual(genome []*gene) *individual {
	return &individual{genome: clone(genome)}
}

// withOperator appends a new ev. operator to the individual.
// All rates should sum to 1.0
func (i *individual) withOperator(op operator, rate float32) *individual {
	i.ops, i.rates = append(i.ops, op), append(i.rates, rate)
	return i
}

// withConstraint appends a new optimization constraint to the individual.
func (i *individual) withConstraint(c constraint) *individual {
	i.consts = append(i.consts, c)
	return i
}

// selectOperator returns a next operator to apply.
// Every operator has an assigned probability, the function selects the operator
// based on roulette selection.
func (i *individual) selectOperator() operator {
	f := randFloat32()

	f0 := float32(0.0)
	for j, r := range i.rates {
		f1 := f0 + r
		if f < f1 {
			return i.ops[j]
		}
		f0 = f1
	}

	return i.ops[randIntn(len(i.ops))]
}

// selectGenome returns a clone of currently the most optimal genome.
// This is kind of (1 + 1) strategy.
func (i *individual) selectGenome() []*gene {
	return clone(i.genome)
}

// optimize tries to find the most optimal genome for a certain amount of time (depends on context),
// or if the solution did not improve for maxiter iterations.
// The function returns the best evaluation and an error (if the context was cancelled).
func (i *individual) optimize(ctx context.Context, maxiter int) (int, error) {
	min := eval(i.genome)
	n := 0
loop:
	for n < maxiter {
		select {
		case <-ctx.Done():
			break loop

		default:
			op := i.selectOperator()
			g, ok := op(i.selectGenome())

			if !ok || !feasible(g, i.consts...) {
				continue
			}

			if v := eval(g); v < min {
				i.genome, min = g, v
				// found a new minimum - reset number of iterations
				n = 0
			} else {
				n++
			}
		}
	}

	return min, ctx.Err()
}

// binTree is a regular binary tree representation
type binTree struct {
	node  *gene
	n     int
	left  *binTree
	right *binTree
}

// newBinTree builds a new binary tree from genome (x-sequence)
func newBinTree(genome []*gene) *binTree {
	var build func() *binTree

	build = func() *binTree {
		if len(genome) == 0 || genome[0].isEmpty() {
			bt := &binTree{n: 1}
			if genome[0] != nil {
				// miss node
				bt.node = &gene{
					weight: genome[0].weight,
				}
			}
			return bt
		}
		bt := &binTree{
			node: &gene{
				key:    genome[0].key,
				weight: genome[0].weight,
			},
			n: 1,
		}

		genome = genome[1:]
		bt.left = build()

		genome = genome[1:]
		bt.right = build()

		bt.n += bt.left.n + bt.right.n
		return bt
	}

	return build()
}

// optimize modifies a binary search tree (bt) such that the total cost of all
// searches is as small as possible.
// optimize returns an optimal (minimum) value.
// The algorithm is deterministic and the complexity is O(n^3).
func (bt *binTree) optimize() int {
	var (
		inorder func(root *binTree)
		p       []*gene    // hit nodes
		q       []*gene    // miss nodes
		n       = bt.n / 2 // number of internal nodes
	)
	p = append(p, nil)
	// inorder inits p, q: p[1] < p[2] < ... < p[n], q[0] < ... < q[n]
	inorder = func(root *binTree) {
		if root == nil {
			return
		}
		inorder(root.left)
		if root.node.isEmpty() {
			if root.node == nil {
				q = append(q, &gene{})
			} else {
				q = append(q, root.node)
			}

		} else {
			p = append(p, root.node)
		}
		inorder(root.right)
	}
	inorder(bt)

	// precalculated optimal values - the total cost will be in e[1][n]
	e := make([][]int, n+2)
	w := make([][]int, n+2)
	// precalculated optimal roots
	root := make([][]int, n)
	for i := 0; i <= n+1; i++ {
		if i < n+2 {
			e[i] = make([]int, n+1)
			w[i] = make([]int, n+1)
		}
		if i < n {
			root[i] = make([]int, n)
		}
		if i > 0 {
			e[i][i-1] = q[i-1].weight
			w[i][i-1] = q[i-1].weight
		}
	}
	for l := 1; l <= n; l++ {
		for i := 1; i+l <= n+1; i++ {
			j := i + l - 1
			e[i][j] = math.MaxInt32
			w[i][j] = w[i][j-1] + p[j].weight + q[j].weight
			for k := i; k <= j; k++ {
				v := e[i][k-1] + e[k+1][j] + w[i][j]
				if v < e[i][j] {
					// found a new minimum
					e[i][j] = v
					root[i-1][j-1] = k
				}
			}
		}
	}

	// rebuild a new optimal binary search tree
	var build func(li, ri int) *binTree
	build = func(li, ri int) *binTree {
		if li < 1 || li > ri || ri > n {
			return &binTree{n: 1}
		}

		r := root[li-1][ri-1]
		t := &binTree{
			node: p[r],
			n:    1,
		}

		t.left = build(li, r-1)
		if t.left.node.isEmpty() {
			t.left.node = q[r-1]
		}

		t.right = build(r+1, ri)
		if t.right.node.isEmpty() {
			t.right.node = q[r]
		}

		t.n += t.left.n + t.right.n
		return t
	}
	*bt = *build(1, n)

	return e[1][n]
}

// genome returns the binary tree (bt) as a genome (x-sequence)
func (bt *binTree) genome() []*gene {
	if bt.node == nil {
		return nil
	}

	var g []*gene
	g = append(g, &gene{
		key:    bt.node.key,
		weight: bt.node.weight,
	})

	if bt.left != nil {
		if l := bt.left.genome(); l != nil {
			g = append(g, bt.left.genome()...)
		} else {
			g = append(g, nil)
		}
	}

	if bt.right != nil {
		if r := bt.right.genome(); r != nil {
			g = append(g, r...)
		} else {
			g = append(g, nil)
		}
	}

	return g
}

// String returns a binary tree in "pretty format"
func (bt *binTree) String() string {
	var build func(root *binTree, prefix string) string

	build = func(root *binTree, prefix string) string {
		if root == nil {
			return ""
		}

		str := prefix
		if len(prefix) > 0 {
			str += "-"
		}

		name := fmt.Sprintf("%s", root.node)
		space := strings.Repeat(" ", len(name)/2)
		if len(prefix) > 0 {
			space += " "
		}

		return str + name + "\n" +
			build(root.right, prefix+space+"|") +
			build(root.left, prefix+space+" ")
	}

	return build(bt, "")
}
