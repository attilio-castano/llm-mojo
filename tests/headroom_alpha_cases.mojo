"""Independent controls for structural alpha before raw Nat reduction.

These compare syntax in one fresh state. The mapped-variable cases make
shared node identities adversarial: equality must respect each active binder
mapping rather than accepting an identical index without inspecting its role.
"""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program


def _n[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def headroom_alpha_controls_accepts() -> Bool:
    var p = tk.empty_proof_terms[160]()
    var nat = _n(p, tk.PT_NAT)
    var zero = _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var successor = _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1)
    var x = _n(p, tk.PT_VAR, nat)
    var y = _n(p, tk.PT_VAR, nat)
    var xeq = _n(p, tk.PT_EQ, x, x)
    var yeq = _n(p, tk.PT_EQ, y, y)
    var hx = _n(p, tk.PT_VAR, xeq)
    var hy = _n(p, tk.PT_VAR, yeq)
    var vx = _n(p, tk.PT_VAR, xeq)
    var vy = _n(p, tk.PT_VAR, yeq)
    # Renaming a dependent binder must rename its domain and body together.
    var pi_x = _n(p, tk.PT_PI, x, _n(p, tk.PT_PI, hx, xeq))
    var pi_y = _n(p, tk.PT_PI, y, _n(p, tk.PT_PI, hy, yeq))
    var lam_x = _n(p, tk.PT_LAM, x, _n(p, tk.PT_LAM, hx, hx))
    var lam_y = _n(p, tk.PT_LAM, y, _n(p, tk.PT_LAM, hy, hy))
    var wrong_domain = _n(p, tk.PT_LAM, x, _n(p, tk.PT_LAM, hy, hy))

    var q = _n(p, tk.PT_VAR, nat)
    var motive = _n(p, tk.PT_LAM, q, _n(p, tk.PT_EQ, q, q))
    var subst_x = _n(p, tk.PT_SUBST, motive, hx, vx)
    var subst_y = _n(p, tk.PT_SUBST, motive, hy, vy)
    var subst_other_evidence = _n(p, tk.PT_SUBST, motive, hx, hx)
    var closed_subst_x = _n(p, tk.PT_LAM, x,
        _n(p, tk.PT_LAM, hx, _n(p, tk.PT_LAM, vx, subst_x)))
    var closed_subst_y = _n(p, tk.PT_LAM, y,
        _n(p, tk.PT_LAM, hy, _n(p, tk.PT_LAM, vy, subst_y)))

    var predecessor = _n(p, tk.PT_VAR, nat)
    var accumulator = _n(p, tk.PT_VAR, nat)
    var identity_step = _n(p, tk.PT_LAM, predecessor,
        _n(p, tk.PT_LAM, accumulator, accumulator))
    var successor_step = _n(p, tk.PT_LAM, predecessor,
        _n(p, tk.PT_LAM, accumulator, _n(p, tk.PT_APP, successor, accumulator)))
    var identity_rec = _n(p, tk.PT_NAT_REC, x, zero, identity_step)
    var successor_rec = _n(p, tk.PT_NAT_REC, x, zero, successor_step)

    var program = make_finishing_program(1, 1, 1)
    var state = tk._make_state[160, 512](program, p)
    if not tk._check_inductive_environment(state):
        return False
    if (not tk._raw_numeral_alpha(state, pi_x, pi_y)
        or not tk._raw_numeral_alpha(state, lam_x, lam_y)
        or not tk._raw_numeral_alpha(state, closed_subst_x, closed_subst_y)
        or not tk._convert(state, pi_x, pi_y)
        or not tk._convert(state, lam_x, lam_y)
        or not tk._convert(state, closed_subst_x, closed_subst_y)):
        return False
    if (tk._raw_numeral_alpha(state, x, y)
        or tk._raw_numeral_alpha(state, lam_x, wrong_domain)
        or tk._raw_numeral_alpha(state, subst_x, subst_other_evidence)
        or tk._raw_numeral_alpha(state, identity_rec, successor_rec)
        or tk._convert(state, subst_x, subst_other_evidence)
        or tk._convert(state, identity_rec, successor_rec)):
        return False

    # Identical x denotes the first bound variable on one side and the second
    # on the other. A node-identity fast path inside mapped recursion is wrong.
    var left_names: List[Int] = [x, y]
    var right_names: List[Int] = [y, x]
    if (tk._raw_numeral_alpha_under(state, x, x, left_names, right_names, 2)
        or tk._equal_under(state, x, x, left_names, right_names, 2)):
        return False
    # Inferring a shared compound node does not close its free variable.
    # Closed compounds may reuse identity under any active mapping; open
    # compounds must still traverse that mapping to distinguish their binders.
    var closed = tk._put(state, tk.PT_EQ, zero, zero)
    if (tk._infer(state, xeq) < 0 or state.context_high[xeq] == 0
        or tk._equal_under(state, xeq, xeq, left_names, right_names, 2)
        or tk._infer(state, closed) < 0 or state.context_high[closed] != 0
        or not tk._equal_under(state, closed, closed, left_names, right_names, 2)):
        return False
    # Conversely, different IDs in matching binder positions are alpha equal.
    return (tk._raw_numeral_alpha_under(state, x, y, left_names, right_names, 2)
        and tk._equal_under(state, x, y, left_names, right_names, 2))
