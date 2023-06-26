# MPT 30B inference code using CPU

Run inference on the latest MPT-30B model using your CPU.

![Inference Demo](media/inference-demo.mp4)

## Requirements

I recommend you use docker for this model, it will make everything easier for you. Tested on cuda-11.8.0 with AMD Epyc CPU.

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