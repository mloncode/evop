package mcts

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

	// operator should return a genome and information if passed genome was modified.
	operator func([]*gene) ([]*gene, bool)

	// constraint is kind of a filter which let us eliminate incorrecty encoded trees.
	constraint func([]*gene) bool

	state struct {
		genome  []*gene
		op      operator // action (mutation | crossover | swap | splay left/right)
		parent  *state
		states  []*state
		reward  int
		nvisits int
	}

	mcts struct {
		ops    []operator // actions
		consts []constraint

		s0 *state // 0-state
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

func (s *state) isLeaf() bool {
	return nil == s || 0 == len(s.states)
}

func (s *state) expand(ops []operator) *state {
	s.states = make([]*state, len(ops))
	for i, op := range ops {
		s.states[i] = &state{
			genome: clone(s.genome),
			op:     op,
			parent: s,
		}
	}
	return s
}

func (s *state) propagate() *state {
	for p := s; p.parent != nil; p = p.parent {
		p.parent.nvisits++
		p.parent.reward += p.reward
	}

	return s
}

func (s *state) String() string {
	var build func(st *state, pf string) string

	build = func(st *state, pf string) string {
		if st == nil {
			return ""
		}

		str := pf
		if len(pf) > 0 {
			str += "-"
		}

		name := fmt.Sprintf("(%d/%d)", st.reward, st.nvisits)
		space := strings.Repeat(" ", len(name)/2)
		if len(pf) > 0 {
			space += " "
		}

		str += name + "\n"

		var ss []*state
		for _, si := range st.states {
			if si.reward == 0 || si.nvisits == 0 {
				continue
			}
			ss = append(ss, si)
		}

		n := len(ss)
		for i, si := range ss {
			if i == n-1 && n > 1 {
				str += build(si, pf+space+" ")
			} else {
				str += build(si, pf+space+"|")
			}
		}

		return str
	}

	return build(s, "")
}

func newMCTS(genome []*gene) *mcts {
	return &mcts{
		s0: &state{
			genome: clone(genome),
		},
	}
}

func (m *mcts) withOperator(op operator) *mcts {
	m.ops = append(m.ops, op)
	return m
}

func (m *mcts) withConstraint(c constraint) *mcts {
	m.consts = append(m.consts, c)
	return m
}

func (m *mcts) optimize(ctx context.Context, maxiter int) (int, error) {
	min := eval(m.s0.genome)
	m.s0.expand(m.ops)

	for n := 0; n < maxiter; n++ {
		select {
		case <-ctx.Done():
			return min, ctx.Err()

		default:
			if s := m.optimizeState(ctx, m.s0, maxiter); s.reward != 0 && s.reward < min {
				min = s.reward
				m.s0.genome = s.genome
				n = 0
			}
		}
	}

	return min, nil
}

func (m *mcts) optimizeState(ctx context.Context, s *state, maxiter int) *state {
	if s == nil {
		s = &state{genome: clone(m.s0.genome)}
	}

	for !s.isLeaf() {
		s = m.selectState(s.states)
	}

	// leaf
	if s.nvisits != 0 {
		s = s.expand(m.ops).states[0]
	}

	return m.rollout(ctx, s, maxiter).propagate()
}

func (m *mcts) selectState(states []*state) *state {
	c := len(states)
	if c == 0 {
		return nil
	}

	maxi := 0
	max := ucb1(float64(-states[maxi].reward), c, m.s0.nvisits, states[maxi].nvisits)
	for i := 1; i < c; i++ {
		s := states[i]
		v := ucb1(float64(-s.reward), c, m.s0.nvisits, s.nvisits)
		if max < v {
			maxi, max = i, v
		}
	}
	return states[maxi]
}

func (m *mcts) rollout(ctx context.Context, s *state, maxiter int) *state {
	s.nvisits++
	g := clone(s.genome)
loop:
	for n := 0; n < maxiter; n++ {
		select {
		case <-ctx.Done():
			break loop

		default:
			g, ok := s.op(g)
			if !ok || !feasible(g, m.consts...) {
				break loop
			}

			r := eval(g)
			if s.reward != 0 && s.reward <= r {
				continue
			}

			n = 0
			s.reward = r
			s.genome = clone(g)
		}
	}

	return s
}

func (m *mcts) String() string {
	return m.s0.String()
}

func ucb1(vi float64, c, n, ni int) float64 {
	if n == 0 || ni == 0 {
		return math.MaxFloat64
	}

	return vi + float64(c)*math.Sqrt(math.Log(float64(n))/float64(ni))
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
