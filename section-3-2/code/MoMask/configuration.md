## Configuration

- Python's version: **3.10**
- Create and enable a virtual enviroment, for instance `.venv`;
- Install the depencencies:
  - Install `wheel` with the following command: `pip install wheel`
  - Install all the depencencies from the `requirements.txt` file;
  - Install `torch`, `torchvision` and `torchaudio` depending of your CUDA version. A possible installation command could be `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`;
- Download MoMask's checkpoints and Glove:
  - Create a folder `checkpoints` inside the root directory of the project;
  - Follow the installation instructions from [Momask's original repository](https://github.com/EricGuo5513/momask-codes). Note that Glove's folder must be placed in the root directory of the project;
- Use `copy_custom_data.py` file to copy all processed file;
- Starting the project:
  - `run.sh` is used to perform inference;
  - `run-train-vq.sh`, `run-train-t2m.sh` and `run-train-res.sh` allow either full retraining (when the `--is_continue` flag is omitted) or finetuning (when the `--is_continue` flag is provided). Each file operate, respectively, on the tokenizer, M-Transformer and R-Transformer;
  - `run-eval-t2m-vq.sh` and `run-eval-t2m-trans-res.sh` handle model evaluation operating, respectively, on the tokenizer and on both transformers.
