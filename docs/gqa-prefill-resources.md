# GQA prefill resource study

This bounded follow-up keeps the original 32x32 MMA kernel (route 8) as its
control. The numerical contract, tile size, head ownership, dependencies and
FP32 accumulation remain fixed. Five individual ablations are implemented:

| Route | Schedule | Question |
| --- | --- | --- |
| 11 | Eight two-value accumulator fragments | Does representation improve compiler lowering? |
| 12 | Rolled QK reduction | Does a shorter instruction schedule reduce live temporary state? |
| 13 | Score barrier removed | Is a block barrier useful for lane-owned score storage? |
| 14 | Probability barrier removed | Is a block barrier useful for lane-owned probability storage? |
| 15 | Lane-owned scores | Does removing the shared score tile offset added live registers? |

The original `_mma` body remains unchanged. The new schedules have a separate
kernel so the control's source does not silently acquire an ablation.

## Ownership proof for the barrier ablations

For each SIMD group, `fr` maps four lanes to each of eight rows. `fc` maps those
four lanes to column starts 0, 2, 4 and 6. Each lane writes and reads its own
`[local_r, 8*j+fc+c]` score and probability locations. No other SIMD group uses
those rows, and no other lane uses those columns. Softmax's cross-lane exchange
is an explicit register shuffle; MMA exchanges already-loaded register
fragments. Therefore these two shared arrays have no cross-thread memory
handoff. Their barriers are tested separately. This argument does not rely on
implicit warp lockstep or passing numerical tests.

K/V are different: cooperative loading and subsequent matrix accesses cross
SIMD groups. The barrier after loading K/V publishes the tile; the final
barrier prevents its reuse while another group is still consuming it. Both
remain in every candidate, with uniform participation including ragged rows.

The register-score candidate retains the original barrier schedule, isolating
storage from synchronization. It increases live score state by eight FP32
values per lane; source arrays and IR allocations are not physical spills.

## Gates and budgets

Every route passes all 29 independent prefill cases, with unchanged tolerance,
plus the causal/full-suffix and shape-rejection tests. Normal execution adds
12 repeated poisoned-output launches per new route per oracle case via
`-D PREFILL_REPEAT=12`. Full repository validation precedes source commits.

Screen full R=T of 16, 1024 and 4096 and incremental (R,T) of (16,4096) and
(64,4096), both hot and ring24, using four paired blocks with self-pairs. Freeze
at most two finalists before the existing eleven-workload final matrix. At
most one combined design may join the five individual ablations. Retain every
observation; a gain must exceed both 5% and matching self-pair noise with every
block faster. General replacement additionally requires no demonstrated
regression on the final grid. A negative result completes the study.

At most twelve focused captures, each below 5000 measured dispatches, may
inspect spills, active time and named counters. Profiling is separate from
latency. Existing tagged sources and evidence remain available. Raw compiler
output and traces stay outside Git; the final explanation belongs in the
existing GQA prefill study.

## Initial inspection

The pinned compiler's host `--emit asm` output did not include device sidecars
for this runtime compilation path. A direct launch with `dump_llvm=True` and
`dump_asm=True` emitted Metal LLVM IR for `_mma[32,32,1]`. It contains a live
`<16 x float>` output value, four barrier calls, and stack allocations around
matrix-primitive argument lowering. This is evidence about intermediate
lowering; it does not identify the physical values responsible for the
previous 144-byte compiler spill event. The final Metal allocator remains
observable through Instruments statistics rather than source-level attribution.

The isolated score/probability ablations each emit 4238 IR lines and three
barrier calls versus the control's 4239 lines and four calls. Tuple output
fragments expose eight two-value loop-carried accumulators and reduce IR
`alloca` sites from 30 to 6. The rolled QK reduction emits 2967 lines, while
lane-owned scores remove the 4 KiB shared score allocation. These counts
confirm implementation differences; none is a performance verdict.
