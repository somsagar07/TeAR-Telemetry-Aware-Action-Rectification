# Release and compatibility notes

The project is now **TeAR: Telemetry-Aware Action Rectification**. The GitHub URL remains `https://github.com/somsagar07/TAM`.

The release keeps the existing research workflow:

- Adapter: `thermal_adapters/tam_bot.py`, class `TAMBoT`.
- Training: `thermal_adapters/train_sft_rl_kl_bct.py`.
- Evaluation: `scripts/eval_transfer_any_base.py`.
- Hardware: `so101/tam_deploy/tam_runtime.py`, class `SO101TAM`.

Checkpoint tensor keys, `tam_*` configuration fields, and `--tam-*` arguments are unchanged. Keep `config.json` beside `sft_adapter.pt`; the evaluator uses it for settings that tensor shapes cannot identify. The README supplies the reference training flags explicitly, since the broader research scripts retain their historical defaults.

The paired mismatch protocol is in `experiments/mismatch/`. Its result keys `tam` and `tam_dr` retain their historical names; reports display TeAR and DR-TeAR. Frozen manifests identify the source and checkpoint files used for each run. Generate a new manifest after changing either.

The feedback baselines now estimate capacity using the final command issued to the degradation model, including the baseline's own correction. Earlier versions used the pre-correction command. Rerun these baselines for results from the corrected implementation; this change does not alter the reference TeAR model or the paired five-method mismatch protocol.

The simulation model and the five-joint SO-101 model have different interfaces. Their checkpoints are not interchangeable. The standalone robot runtime keeps its existing filenames.
