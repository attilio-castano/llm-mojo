# Benchmarks

Benchmark programs are reusable measurement instruments, not experiment
records. The project's [experimental method](../docs/experiments.md) defines how
to declare a hypothesis, retain runs, separate profiling from timing, and bound
the resulting claims.

The RMSNorm benchmark measures the supported Apple GPU enqueue path for two
BF16 workloads: one `1 x 896` row representative of decode and `128 x 896`
rows representative of a prefill operation. Device allocation, initialization,
and host transfers are outside the timed region. Each workload first runs an
untimed all-ones correctness gate.

Run it through the metadata wrapper:

```bash
uv run --locked python benchmarks/run_rms_norm.py
```

The wrapper prints the commit and dirty state, curated hardware and software
identity, power and thermal conditions, available memory, attached display
configuration, and then the benchmark result. It deliberately excludes serial
numbers and hardware UUIDs. Capture the complete stdout outside the repository
during exploration. A recorded experiment retains the compact protocol, raw
timing samples, provenance, and conclusion as described by the experimental
method; large generated traces remain outside Git and are referenced by
checksum.

The reported byte throughput is derived from logical BF16 input, weight, and
output traffic per row. It is useful for comparing identical runs of this
kernel; it is not a direct measurement of DRAM traffic or memory bandwidth.
