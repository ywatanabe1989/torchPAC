<!-- ---
!-- title: ./torchPAC/scripts/Handlers/README.md
!-- author: ywatanabe
!-- date: 2024-11-04 21:34:36
!-- --- -->


| Variable | Tensorpac | MNGS (torchPAC) | Impact | Tensorpac Ref | MNGS Ref |
|----------|-----------|-----------------|---------|---------------|-----------|
| `chunk_size` | ✓ | ✓ | Memory usage & speed | [TensorpacHandler.py#L118-L121](TensorpacHandler.py#L118-L121) | [MNGSHandler.py#L65-L76](MNGSHandler.py#L65-L76) |
| `fp16` | ✓ | ✓ | Precision & memory | [TensorpacHandler.py#L40](TensorpacHandler.py#L40) | [MNGSHandler.py#L62](MNGSHandler.py#L62) |
| `no_grad` | ✘ | ✓ | Training efficiency | - | [MNGSHandler.py#L48](MNGSHandler.py#L48) |
| `in_place` | ✘ | ✓ | Memory optimization | - | [MNGSHandler.py#L48](MNGSHandler.py#L48) |
| `trainable` | ✘ | ✓ | Filter adaptability | - | [MNGSHandler.py#L49](MNGSHandler.py#L49) |
| `use_threads` | ✓ | ✘ | CPU parallelization | [TensorpacHandler.py#L119](TensorpacHandler.py#L119) | - |
| `device` | ✘ | ✓ | Hardware selection | - | [MNGSHandler.py#L50](MNGSHandler.py#L50) |
| `batch_size` | ✓ | ✓ | Processing throughput | [TensorpacHandler.py#L117](TensorpacHandler.py#L117) | [MNGSHandler.py#L61](MNGSHandler.py#L61) |
| `n_chs` | ✓ | ✓ | Channel load | [TensorpacHandler.py#L117](TensorpacHandler.py#L117) | [MNGSHandler.py#L61](MNGSHandler.py#L61) |
| `n_segments` | ✓ | ✓ | Time windows | [TensorpacHandler.py#L117](TensorpacHandler.py#L117) | [MNGSHandler.py#L61](MNGSHandler.py#L61) |
| `fs` | ✓ | ✓ | Frequency resolution | [TensorpacHandler.py#L38](TensorpacHandler.py#L38) | [MNGSHandler.py#L42](MNGSHandler.py#L42) |
| `pha/amp_n_bands` | ✓ | ✓ | Frequency bands | [TensorpacHandler.py#L39](TensorpacHandler.py#L39) | [MNGSHandler.py#L43-L46](MNGSHandler.py#L43-L46) |
| `n_perm` | ✓ | ✓ | Statistics | [TensorpacHandler.py#L41](TensorpacHandler.py#L41) | [MNGSHandler.py#L47](MNGSHandler.py#L47) |

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
