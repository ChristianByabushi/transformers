# HW4 — Transformers

## HW4P1: Autoregressive Language Modeling with a Causal Transformer Decoder

### Model
Decoder-only Transformer (GPT-style, Pre-LN). Single stack of `SelfAttentionDecoderLayer` blocks, each containing masked multi-head self-attention and a position-wise feedforward network. Causal masking prevents attending to future tokens. Optional weight tying between the token embedding and the final linear projection.

### Training Strategy
- Optimizer: AdamW
- Scheduler: Cosine annealing with warmup
- Loss: CrossEntropyLoss with label smoothing, padding tokens ignored
- Gradient accumulation for effective large batch training
- Mixed precision (fp16) via GradScaler

### Augmentations
None — language modeling uses raw tokenized text sequences.

### Notebook Execution
1. Ensure the working directory is the handout root (`IDL-HW4/`)
2. Run all cells in `HW4P1_nb.ipynb` sequentially
3. Data should be at `hw4_data/hw4p1_data/`
4. Checkpoints saved to `checkpoint-best-metric-model.pth`

---

## HW4P2: Automatic Speech Recognition with an Encoder-Decoder Transformer

### Model
Pre-LN Encoder-Decoder Transformer for ASR. The encoder processes 80-dimensional filterbank features through a `SpeechEmbedding` layer (CNN/LSTM-based downsampling) followed by `SelfAttentionEncoderLayer` blocks. The decoder uses `CrossAttentionDecoderLayer` blocks with masked self-attention and cross-attention to the encoder output. A CTC auxiliary head on the encoder output provides additional training signal. Optional weight tying between target embedding and final linear layer.

### Training Strategy
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau on validation CER
- Loss: Joint CE + CTC loss (`ctc_weight` controls the balance)
- Label smoothing on CE loss
- Gradient accumulation for effective large batch training
- Mixed precision (fp16) via GradScaler
- SpecAugment data augmentation during training

### Augmentations
SpecAugment applied in `collate_fn` on padded batches:
- Frequency masking: masks random frequency bands
- Time masking: masks random time steps
- Global MVN normalization on filterbank features

### Notebook Execution
1. Ensure the working directory is the handout root (`IDL-HW4/`)
2. Run all cells in `HW4P2_nb.ipynb` sequentially
3. Data should be at `hw4_data/hw4p2_data/`
4. Training set global stats (mean/std) are computed automatically and passed to val/test datasets
5. After training, run the evaluation cell to generate `submission.csv` for Kaggle
6. Checkpoints saved to `checkpoint-best-metric-model.pth`
