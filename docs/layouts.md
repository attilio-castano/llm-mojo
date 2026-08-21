# Layout language

This project uses a small, concrete language to describe how tensor operations
map onto memory and hardware. The language separates four questions that are
easy to conflate:

1. **Value:** What are the logical axes, extents, dtypes, and arithmetic?
2. **Storage:** Where does each logical coordinate live in linear memory?
3. **Work:** Which CPU iteration, SIMD lane, or GPU thread owns each element?
4. **Order:** In what order are reductions performed, where do casts occur, and
   what synchronization makes intermediate values visible?

A storage layout does not answer the work or order questions. Likewise, a
thread partition does not establish a numerical reduction order or prove which
device executed it.

## Storage notation

Write a strided layout as:

```text
shape : stride
```

For a two-dimensional layout `(R, H) : (sr, sh)`, logical coordinate `(r, h)`
maps to linear offset:

```text
r * sr + h * sh
```

For example, `(R, H) : (H, 1)` is row-major storage with a contiguous hidden
axis. A zero stride denotes a projected or broadcast view: `(R, H) : (0, 1)`
reuses one length-`H` row for every logical row without materializing copies.

The notation describes an address mapping, not an allocation. Two views may
share storage, and an implementation must state which object owns that storage.

## RMSNorm V0

Let `R` be the number of token rows and let `H = 896` be the Qwen hidden size.
The operation has these logical tensors:

```text
X[row, hidden]  BF16  input activations
W[hidden]       BF16  learned weights
Y[row, hidden]  BF16  output activations
```

The initial storage contract is:

| View | Layout | Meaning |
| --- | --- | --- |
| `X` | `(R, H) : (H, 1)` | Contiguous hidden values for each row |
| `Y` | `(R, H) : (H, 1)` | Same physical organization as `X` |
| `W` as seen by `Y` | `(R, H) : (0, 1)` | One weight row projected across all token rows |
| inverse RMS as seen by `Y` | `(R, H) : (1, 0)` | One scalar per row projected across the hidden axis |

The projected views are semantic descriptions; the reference implementation
need not construct them as objects or allocate their repeated values.

The reference work mapping is one explicit row traversal followed by one
explicit hidden-axis traversal. A GPU implementation must separately record
its row tile, vector width, thread or SIMD-group layout, reduction tree,
synchronization boundaries, and actual backend identity.

## Use in code and evidence

- Name semantic axes before reducing them to integer positions.
- Record logical shape, physical strides, dtype, and ownership at operation
  boundaries.
- Record work partition and reduction schedule separately from storage layout.
- Use Mojo's native layout and tensor-view facilities when they express the
  required mapping; do not build a project-specific layout algebra.
- Treat tiling, vectorization, projection, and composition as views until an
  implementation proves that data moved or was allocated.
- Do not infer performance from a layout. Performance evidence requires a
  synchronized benchmark and verified device/backend identity.
- Add vocabulary only when implemented code creates a distinction that needs a
  name.

This language is informed by the shape/stride layout algebra described in
[Categorical Foundations for CuTe Layouts](https://arxiv.org/abs/2601.05972),
while remaining tied to the operations and native Mojo facilities used here.
