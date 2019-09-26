package evop

import (
	"context"
	"fmt"
	"math"
	"strings"
)

type (
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
		c  int    // exploration const
	}
)

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
		c: 5,
	}
}

func (m *mcts) withExploration(c int) *mcts {
	m.c = c
	return m
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
	maxi := 0
	max := ucb1(-states[maxi].reward, m.c, m.s0.nvisits, states[maxi].nvisits)
	for i := 1; i < len(states); i++ {
		s := states[i]
		v := ucb1(-s.reward, m.c, m.s0.nvisits, s.nvisits)
		if max < v {
			maxi, max = i, v
		}
	}
	return states[maxi]
}

func (m *mcts) rollout(ctx context.Context, s *state, maxiter int) *state {
	s.nvisits++
	s.reward = eval(s.genome)
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

func ucb1(vi, c, n, ni int) float64 {
	if n == 0 || ni == 0 {
		return math.MaxFloat64
	}

	fvi, fc, fn, fni := float64(vi), float64(c), float64(n), float64(ni)
	return fvi/fni + fc*math.Sqrt(math.Log(fn)/fni)
}
