# 📘 HW4 Handout

This assignment introduces a new format designed to improve your development workflow. The main goals are:

- **Test Suite Integration**: Your code will be tested similarly to HW1.
- **Local Development**: You can perform most development locally, reducing compute usage.
- **Hands-on Experience**: Build a full deep learning pipeline with fewer abstractions than before.

---

## ⚠️ Important: Working Directory

For the provided notebooks to work, your **current working directory must match the handout directory**.

This is required because all imports use **relative paths**.

### You can fix this by:
1. Moving your notebook into the handout directory  
2. OR changing the working directory in Python:
```python
import os
os.chdir("path/to/handout")


.
├── README.md
├── hw4lib/
├── mytorch/
├── tests/
├── hw4_data_subset/
└── requirements.txt

hw4_data_subset/
├── hw4p1_data/          # Causal Language Modeling
│   ├── train/
│   ├── valid/
│   └── test/
└── hw4p2_data/          # Speech Recognition
    ├── dev-clean/
    │   ├── fbank/
    │   └── text/
    ├── test-clean/
    │   └── fbank/
    └── train-clean-100/
        ├── fbank/
        └── text/

hw4lib/
├── data/
│   ├── tokenizer_jsons/
│   ├── asr_dataset.py
│   ├── lm_dataset.py
│   └── tokenizer.py
├── decoding/
│   └── sequence_generator.py
├── model/
│   ├── masks.py
│   ├── positional_encoding.py
│   ├── speech_embedding.py
│   ├── sublayers.py
│   ├── decoder_layers.py
│   ├── encoder_layers.py
│   └── transformers.py
├── trainers/
│   ├── base_trainer.py
│   ├── asr_trainer.py
│   └── lm_trainer.py
└── utils/
    ├── create_lr_scheduler.py
    └── create_optimizer.py

mytorch/nn/
├── activation.py
├── linear.py
├── scaled_dot_product_attention.py
└── multi_head_attention.py


tests/
├── testing_framework.py
├── test_mytorch*.py
├── test_dataset*.py
├── test_mask*.py
├── test_positional_encoding.py
├── test_sublayers*.py
├── test_encoderlayers*.py
├── test_decoderlayers*.py
├── test_transformers*.py
├── test_hw4p1.py
└── test_decoding.py
