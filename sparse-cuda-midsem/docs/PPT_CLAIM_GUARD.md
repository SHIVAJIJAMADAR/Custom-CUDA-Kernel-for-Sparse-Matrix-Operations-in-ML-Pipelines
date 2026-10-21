# PPT Claim Guard

Use the following rule before presenting the mid-sem deck:

- If a statement says **implemented**, there should be a corresponding source file and a successful local/GPU execution where relevant.
- If a statement says **verified**, keep the test output/log that demonstrates verification.
- If a statement says **benchmark established**, keep the raw benchmark CSV/JSON and environment information.
- If a statement says **outperforms cuSPARSE**, keep reproducible benchmark output showing the exact matrix, batch, dtype, format, warm-up and timing protocol.

Do not use the demo JSON in `results/demo_profile.json` as measured performance evidence.
