# MPT 30B inference code using CPU

Run inference on the latest MPT-30B model using your CPU. This inference code uses a [ggml](https://github.com/ggerganov/llama.cpp) quantized model. To run the model we'll use a library called [ctransformers](https://github.com/marella/ctransformers) that has bindings to ggml in python.

I recommend a system with 32GB of ram.

[Inference Demo](https://github.com/abacaj/mpt-30B-inference/assets/7272343/486fc9b1-8216-43cc-93c3-781677235502)

## Requirements

I recommend you use docker for this model, it will make everything easier for you. Tested on AMD Epyc CPU.

## Setup

First create a venv.

```sh
python -m venv env && source env/bin/activate
```

Next install dependencies.

```sh
pip install -r requirements.txt
```

Next download the quantized model weights (about 19GB).

```sh
python download_model.py
```

Ready to rock, run inference.

```sh
python inference.py
```

Next modify inference script prompt and generation parameters.
