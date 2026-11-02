# Apple validation evidence

These files distinguish source parity, Core ML numerical parity, app builds and
physical-device inference. Each numerical report identifies its hardware,
compute-unit choice, model profile, date and runtime. Mac CPU correctness timings
under co-resident conversion load are not GPU or iPhone FPS benchmarks.

Intermediate failed reports are retained when they explain a required precision
change. Only the report for the current package hashes applies to the delivered
model. A pass on synthetic weights is insufficient to certify the real PRO model.
Device inference and sustained animation quality are separate gates from these
component checks.

The [runtime performance report](../PERFORMANCE.md) links paired Mac runs, full-frame comparisons and stage-by-stage speech backend checks.
