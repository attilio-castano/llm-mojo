"""Checked Base equations and focused caller-side rejection controls."""
from std.collections import Array
import llm_mojo.kernels.proof_term_kernel as tk


def base_driver_equations_accepts(stride: Int = 17, mode: Int = -1) -> Bool:
    # Shape only chooses the theorem's stride argument. Both q and the body
    # of the generic Base definition remain symbolic during type checking.
    var p = tk.empty_proof_terms[96]()
    var nat = tk.pt_add(p, tk.PT_NAT)
    var zero = tk.pt_add(p, tk.PT_LITERAL, value=0)
    var one = tk.pt_add(p, tk.PT_LITERAL, value=1)
    var increment = tk.pt_add(p, tk.PT_LITERAL, value=stride)
    var q = tk.pt_add(p, tk.PT_VAR, nat)
    var successor = tk.pt_add(p, tk.PT_ADD, q, one)
    var base_zero = tk.pt_add(p, tk.PT_BASE, zero)
    var base_q = tk.pt_add(p, tk.PT_BASE, q)
    var zero_proof = tk.pt_add(p, tk.PT_ARITHMETIC, value=tk.DEC_BASE_ZERO)
    var step_declaration = tk.pt_add(p, tk.PT_ARITHMETIC, value=tk.DEC_BASE_STEP)
    var step_proof = tk.pt_add(p, tk.PT_LAM, q,
        tk.pt_add(p, tk.PT_APP, step_declaration, q))
    var zero_goal = tk.pt_add(p, tk.PT_EQ, base_zero, zero)
    var step_goal = tk.pt_add(p, tk.PT_PI, q,
        tk.pt_add(p, tk.PT_EQ, tk.pt_add(p, tk.PT_BASE, successor),
            tk.pt_add(p, tk.PT_ADD, base_q, increment)))
    if mode == 0:
        # Specialization cannot silently capture a different program stride.
        var wrong = tk.pt_add(p, tk.PT_LITERAL, value=stride + 1)
        step_goal = tk.pt_add(p, tk.PT_PI, q,
            tk.pt_add(p, tk.PT_EQ, tk.pt_add(p, tk.PT_BASE, successor),
                tk.pt_add(p, tk.PT_ADD, base_q, wrong)))
    elif mode == 1:
        var two = tk.pt_add(p, tk.PT_LITERAL, value=2)
        step_goal = tk.pt_add(p, tk.PT_PI, q,
            tk.pt_add(p, tk.PT_EQ, tk.pt_add(p, tk.PT_BASE,
                tk.pt_add(p, tk.PT_ADD, q, two)),
                tk.pt_add(p, tk.PT_ADD, base_q, increment)))
    elif mode == 2:
        # Driver equation exports have no caller-controlled parameter payload.
        _ = tk.pt_add(p, tk.PT_ARITHMETIC, extra=zero, value=tk.DEC_BASE_ZERO)
    elif mode == 3:
        zero_proof = step_declaration
    elif mode == 4:
        # Exercise closed zero conversion through the private checked Base
        # bridge directly, independently of the exported zero signature.
        # Direct successor conversion with huge raw Add literals remains a
        # separate bounded-checker capability. The generic bootstrap checks
        # successor by Refl with symbolic stride; this consumer uses its export.
        zero_proof = tk.pt_add(p, tk.PT_REFL, zero)
    var judgments = Array[tk.ProofTermJudgment, 2](
        fill=tk.ProofTermJudgment(zero_proof, zero_goal))
    judgments[1] = tk.ProofTermJudgment(step_proof, step_goal)
    var program = tk.FinishingProgram(stride, 1, stride, 0, 1, stride, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms_many(program, p, judgments)
