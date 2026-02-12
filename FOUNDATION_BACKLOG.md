# RelGT Foundation Model Backlog / Notes

Goal: Make RelGT closer to a foundation model by pretraining across multiple datasets + tasks.

## Option 1 (current experiment): fast MLP-style table encoder
- Replace TorchFrame ResNet with TorchFrame MLP as the per-table row encoder.
- Intended as a simpler, cheaper backbone to iterate with.
- Added CLI flag: `--tf_model {resnet,mlp}`.

Next (if we pursue true foundation):
- Add self-supervised pretraining objective for the tabular encoder (masked feature modeling / denoising autoencoder).
- Cache row embeddings per table to avoid recompute.

## Option 2 (next candidate): FT-Transformer / tokenized tabular transformer
- Use an FT-Transformer-like encoder (categorical token embeddings + numerical projections + transformer blocks).
- Can be pretrained with masked-feature modeling across datasets.
- Integrate as drop-in replacement for `NeighborTfsEncoder` output vectors.

## TabPFN idea (previous discussion)
- TabPFN embeddings can be extracted via `clf.get_embeddings(X_query)`.
- However, typical usage requires `fit(X, y)` first, so embeddings are conditioned on a label choice.
- Possible workarounds:
  - pseudo-labels per table (self-supervised), then cache embeddings
  - or fit using downstream task labels (less foundation-like)
- Practical blockers observed:
  - gated HF model requires accepting terms on HF
  - CPU sample limits unless overridden; GPU recommended

## General pretraining objectives (for later)
- Masked feature modeling (MFM): mask columns, reconstruct
- Denoising autoencoder (DAE): corruption + reconstruction
- Contrastive (SCARF-style): two corruptions, InfoNCE
- Graph objectives: edge-type prediction, link prediction, temporal contrast

