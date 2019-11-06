package evop

import "fmt"

type (
	gene struct {
		key    string
		weight int
	}

	// operator should return a genome and information if passed genome was modified.
	operator func([]*gene) ([]*gene, bool)

	// constraint is kind of a filter which let us eliminate incorrecty encoded trees.
	constraint func([]*gene) bool
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

func (g *gene) equal(gg *gene) bool {
	if g == nil && gg == nil {
		return true
	}

	if g == nil || gg == nil {
		return false
	}

	return *g == *gg
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
func inversion(genome []*gene) ([]*gene, bool) {
	n := len(genome)
	k1, k2 := randIntn(n), randIntn(n)
	if k1 == k2 {
		return genome, false
	}
	if k1 > k2 {
		k1, k2 = k2, k1
	}

	return inversionAt(genome, k1, k2), true
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
func swap(genome []*gene) ([]*gene, bool) {
	n := len(genome)
	k1, k2 := randNormIntn(n), randNormIntn(n)
	if k1 == k2 {
		return genome, false
	}
	return swapAt(genome, k1, k2), true
}

func swapAt(genome []*gene, k1, k2 int) []*gene {
	genome[k1], genome[k2] = genome[k2], genome[k1]
	return genome
}

// ([a, b, c, d] [e, f], g) -> ([e, f] [a, b, c, d], g)
func crossover(genome []*gene) ([]*gene, bool) {
	n := len(genome)
	k := randIntn(n - 3)
	return crossoverAt(genome, k), true
}

func crossoverAt(genome []*gene, k int) []*gene {
	n := len(genome)
	head, tail := clone(genome[0:k+1]), clone(genome[k+1:n-2])
	copy(genome[0:len(tail)], tail)
	copy(genome[len(tail):], head)

	return genome
}

func splayLeft(genome []*gene) ([]*gene, bool) {
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
				return splayLeftAt(genome, k), true
			}
		}
	}

	return genome, false
}

func splayLeftAt(genome []*gene, k int) []*gene {
	g := genome[k]
	copy(genome[1:k+1], genome[0:k])
	genome[0] = g
	return genome
}

func splayLeftSubTree(genome []*gene) ([]*gene, bool) {
	n := len(genome)
	return splayLeftSubTreeAt(genome, randIntn(n))
}

func splayLeftSubTreeAt(genome []*gene, root int) ([]*gene, bool) {
	n := len(genome)
	k1 := subTreeIndex(genome, root)
	cnt0, cnt1 := 0, 0
	for i := k1; i < n; i++ {
		if genome[i].isEmpty() {
			cnt0++
		} else {
			cnt1++
		}

		if cnt0 > cnt1 {
			k2 := i
			if k2 < n {
				_, b := splayLeft(genome[k1 : k2+1])
				return genome, b
			}
		}
	}

	return genome, false
}

func splayRight(genome []*gene) ([]*gene, bool) {
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
				return splayRightAt(genome, k1, k2), true
			}
		}
	}

	return genome, false
}

func splayRightAt(genome []*gene, k1, k2 int) []*gene {
	g := genome[0]
	copy(genome[0:], genome[k1:k2+1])
	genome[k2] = g
	return genome
}

func splayRightSubTree(genome []*gene) ([]*gene, bool) {
	n := len(genome)
	if n > 1 && genome[1] != nil {
		return splayRightSubTreeAt(genome, randIntn(n))
	}

	return genome, false
}

func splayRightSubTreeAt(genome []*gene, root int) ([]*gene, bool) {
	n := len(genome)
	k1 := subTreeIndex(genome, root)

	cnt0, cnt1 := 0, 0
	for i := k1; i < n; i++ {
		if genome[i].isEmpty() {
			cnt0++
		} else {
			cnt1++
		}

		if cnt0 > cnt1 {
			k2 := i
			if k2 < n {
				_, b := splayRight(genome[k1 : k2+1])
				return genome, b
			}
		}
	}

	return genome, false
}

func subTreeIndex(genome []*gene, root int) int {
	n := 0
	for k := root; k > 0; k-- {
		if genome[k].isEmpty() {
			n++
			continue
		}

		n--
		if n < 2 {
			return k
		}
	}

	return 0
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

func equal(g1 []*gene, g2 []*gene) bool {
	if len(g1) != len(g2) {
		return false
	}

	for i, g := range g1 {
		if !g.equal(g2[i]) {
			return false
		}
	}

	return true
}
