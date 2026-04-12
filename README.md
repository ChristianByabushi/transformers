# TRANSFORMERS

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
