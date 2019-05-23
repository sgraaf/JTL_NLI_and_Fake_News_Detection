# JTL_NLI_and_Fake_News_Detection

## Instructions

In order to run the script, due to some CPU/CUDA handling issues, you need to change 2 scripts in your PyTorch library. 
- File `venv/lib/python3.6/site-packages/torch/nn/functional.py` has to be replaced with the modified file in this repository `src/functional.py`.
- File `venv/lib/python3.6/site-packages/torch/nn/modules/rnn.py` has to be replaced with the modified file in this repository `src/rnn.py`.
