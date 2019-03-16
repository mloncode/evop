package bintree

import (
	"context"
	"fmt"
	"math"
	"strings"
)

type (
	gene struct {
		key    string
		weight int
	}

	operator   func([]*gene) []*gene
	constraint func([]*gene) bool

	individual struct {
		genome []*gene
		ops    []operator
		rates  []float32
		consts []constraint
	}
)

// isEmpty returns true if gene has no information.
// A gene is empty if:
// 1) is nil,
// 2) key is an empty string (no information) -
// it represents a "miss" node in a tree (e.g. we couldn't find a key with certain information).
// "Miss" nodes are also empty genes (leaves), but with weight > 0.
func (g *gene) isEmpty() bool {
	return nil == g || "" == g.key
}

// String returns a gene in "printing friendly" format
func (g *gene) String() string {
	if nil == g {
		return "(nil)"
	}
	if "" == g.key {
		return fmt.Sprintf("(%d)", g.weight)
	}

	return fmt.Sprintf("(%s/%d)", g.key, g.weight)
}

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
			g := op(i.selectGenome())

			if !feasible(g, i.consts...) {
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

// clone returns a copy of given genome.
func clone(genome []*gene) []*gene {
	if genome == nil {
		return nil
	}

	g := make([]*gene, len(genome))
	copy(g, genome)
	return g
}

// feasible tests genome against all constraints.
// If the genome passes the constraints, it is feasible -
// the function returns true, otherwise false.
func feasible(genome []*gene, consts ...constraint) bool {
	for _, c := range consts {
		if !c(genome) {
			return false
		}
	}
	return true
}

// eval evaluates a given genome with fitness function: ∑wi(hi + 1), i in [0, n)
func eval(genome []*gene) int {
	var fn func(int) int

	// fitness function
	fn = func(h int) int {
		if len(genome) == 0 || genome[0] == nil {
			return 0
		}
		if genome[0].isEmpty() {
			// miss node - no information, but it has a weight.
			return h * genome[0].weight
		}

		s := h * genome[0].weight
		genome = genome[1:]
		s += fn(h + 1)

		genome = genome[1:]
		s += fn(h + 1)
		return s
	}

	// start from the root (height: 0 + 1)
	return fn(1)
}

// (a, b, [c, d, e], f, g) -> (a, b, [e, d, c], f, g)
func inversion(genome []*gene) []*gene {
	n := len(genome)
	k1, k2 := randIntn(n), randIntn(n)
	if k1 > k2 {
		k1, k2 = k2, k1
	}
	return inversionAt(genome, k1, k2)
}

func inversionAt(genome []*gene, k1, k2 int) []*gene {
	for k1 < k2 {
		genome[k1], genome[k2] = genome[k2], genome[k1]
		k1++
		k2--
	}
	return genome
}

// (a, [b], c, d, e, [f], g) -> (a, [f], c, d, e, [b], g)
func swap(genome []*gene) []*gene {
	n := len(genome)
	k1, k2 := randNormIntn(n), randNormIntn(n)
	return swapAt(genome, k1, k2)
}

func swapAt(genome []*gene, k1, k2 int) []*gene {
	genome[k1], genome[k2] = genome[k2], genome[k1]
	return genome
}

// ([a, b, c, d] [e, f], g) -> ([e, f] [a, b, c, d], g)
func crossover(genome []*gene) []*gene {
	n := len(genome)
	k := randIntn(n - 3)
	return crossoverAt(genome, k)
}

func crossoverAt(genome []*gene, k int) []*gene {
	n := len(genome)
	head, tail := clone(genome[0:k+1]), clone(genome[k+1:n-2])
	copy(genome[0:len(tail)], tail)
	copy(genome[len(tail):], head)

	return genome
}

func splayLeft(genome []*gene) []*gene {
	n := len(genome)
	cnt0, cnt1 := 0, 0
	for i := 1; i < n; i++ {
		if genome[i].isEmpty() {
			cnt0++
		} else {
			cnt1++
		}

		if cnt0 > cnt1 {
			k := i + 1
			if k < n && !genome[k].isEmpty() {
				return splayLeftAt(genome, k)
			}
		}
	}

	return genome
}

func splayLeftAt(genome []*gene, k int) []*gene {
	g := genome[k]
	copy(genome[1:k+1], genome[0:k])
	genome[0] = g
	return genome
}

func splayRight(genome []*gene) []*gene {
	n := len(genome)
	if n > 1 && genome[1] != nil {
		k1 := 1
		cnt0, cnt1 := 0, 0
		for i := 2; i < n; i++ {
			if genome[i].isEmpty() {
				cnt0++
			} else {
				cnt1++
			}

			if cnt0 > cnt1 {
				k2 := i
				return splayRightAt(genome, k1, k2)
			}
		}
	}

	return genome
}

func splayRightAt(genome []*gene, k1, k2 int) []*gene {
	g := genome[0]
	copy(genome[0:], genome[k1:k2+1])
	genome[k2] = g
	return genome
}

// isBinTree is a constraint which checks if given genome is a correctly encoded binary tree
func isBinTree(genome []*gene) bool {
	n := len(genome) - 1
	cnt0, cnt1 := 0, 0

	for i := 0; i < n; i++ {
		if genome[i].isEmpty() {
			cnt0++
		} else {
			cnt1++
		}

		if cnt0 > cnt1 {
			return false
		}
	}

	return genome[n].isEmpty()
}

// isBST is a constraint which checks if given genome is a correctly encoded binary search tree.
// The function assumes that genome is correctly encoded binary tree,
// and only checks BST constraints (left < root < right).
func isBST(genome []*gene) bool {
	var (
		keys    []string
		inorder func() error
	)

	inorder = func() error {
		if len(genome) == 0 || genome[0].isEmpty() {
			return nil
		}
		root := genome[0].key

		genome = genome[1:]
		if err := inorder(); err != nil {
			return err
		}

		keys = append(keys, root)
		if n := len(keys); n > 1 {
			if keys[n-2] > keys[n-1] {
				return fmt.Errorf("isBST: %s > %s", keys[n-2], keys[n-1])
			}
			keys = keys[1:]
		}

		genome = genome[1:]
		if err := inorder(); err != nil {
			return err
		}

		return nil
	}

	err := inorder()
	return err == nil
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
