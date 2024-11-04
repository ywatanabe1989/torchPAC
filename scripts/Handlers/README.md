<!-- ---
!-- title: ./torchPAC/scripts/Handlers/README.md
!-- author: ywatanabe
!-- date: 2024-11-04 21:53:24
!-- --- -->


| Variable      | Tensorpac | MNGS (torchPAC) | Impact                  n                         | Tensorpac Ref                                                  | MNGS Ref                                             |
|---------------|-----------|-----------------|---------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------|
| `chunk_size`  | ✓         | ✓               | # of Signal Samples calculated at one itereation  | [TensorpacHandler.py#L118-L121](TensorpacHandler.py#L118-L121) | [MNGSHandler.py#L143-L169](MNGSHandler.py#L143-L169) |
| `no_grad`     | ✘         | ✓               | Whether to calculate gradiation (for AI training) | -                                                              | [MNGSHandler.py#L48](MNGSHandler.py#L48) fixme       |
| `in_place`    | ✘         | ✓               | Whether to calculate in the in_place manner       | -                                                              | [MNGSHandler.py#L48](MNGSHandler.py#L96)             |
| `trainable`   | ✘         | ✓               | Whether to use trainable model                    | -                                                              | [MNGSHandler.py#L49](MNGSHandler.py#L97)             |
| `use_threads` | ✓         | ✘               | CPU parallelization                               | [TensorpacHandler.py#L66](TensorpacHandler.py#L66)             | -                                                    |
| `device`      | ✘         | ✓               | Hardware selection (CPU or GPU)                   | -                                                              | [MNGSHandler.py#L50](MNGSHandler.py#L98)             |

| Variable       | Tensorpac | MNGS (torchPAC) | Impact                                               | Tensorpac Ref                                                   | MNGS Ref                                          |
|----------------|-----------|-----------------|------------------------------------------------------|-----------------------------------------------------------------|---------------------------------------------------|
| `batch_size`   | ✓         | ✓               | # of Signal Samples                                  | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)          |
| `n_chs`        | ✓         | ✓               | # of Signal Channels                                 | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)          |
| `n_segments`   | ✓         | ✓               | # of Signal segments                                 | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)          |
| `t_sec`        | ✓         | ✓               | Signal Length for Batch Samples [s]                  |                                                                 |                                                   |
| `fs`           | ✓         | ✓               | Sampling frequency [1/Hz]                            | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)          |
| `pha_n_bands`  | ✓         | ✓               | # of Frequency bands for phase                       | [_BaseHandler.py#L106](_BaseHandler.py#L39)                     | [_BaseHandler.py#L43-L46](BaseHandler.py#L43-L46) |
| `ampt_n_bands` | ✓         | ✓               | # of Frequency bands for amplitude                   | [_BaseHandler.py#L106](_BaseHandler.py#L39)                     | [_BaseHandler.py#L43-L46](BaseHandler.py#L43-L46) |
| `n_perm`       | ✓         | ✓               | # of permutations to calculate z score of PAC values | [TensorpacHandler.py#L41](TensorpacHandler.py#L41)              | [MNGSHandler.py#L47](MNGSHandler.py#L47)          |
| `fp16`         | ✓         | ✓               | Use float16 precision (default float32)              | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)          |

✓: Impacts calculation
✘: No impact/Not applicable

## Batch_size vs. chunk_size
The key equation is:
n_chunks = math.ceil(len(xx) / self.chunk_size)

**batch_size**: Initial data dimension (batch_size, n_chs, n_segments, seq_len)
- Determines total data to process
- Fixed input parameter

## Chunk_size (effective batch size)
**chunk_size**: The effective batch size for computation at an iteration
- Splits batch_size into smaller chunks
- Affects memory-performance trade-off
- Controls effective batch size during computation
- Formula: n_chunks = math.ceil(batch_size / chunk_size)

./scripts/Handlers/MNGSHandler.py#L143-169
./scripts/Handlers/TensorpacHandler.py#L118-121

chunk size does not affect the use_threads mode for Tensorpac calculation
