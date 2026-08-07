# SSC
Official code implementation for ["Spatial Semantic Communication: When Semantic Transmission Meets Index Modulation"](https://ieeexplore.ieee.org/document/11643249)

## Citation
```bibtex
@ARTICLE{11643249,
  author={Guo, Xinghao and Xu, Yin and He, Dazhi and Hong, Hanjiang and Chen, Zhiyong and Zhang, Cixiao and Wu, Yiyan and Zhang, Wenjun},
  journal={IEEE Transactions on Communications}, 
  title={Spatial Semantic Communication: When Semantic Transmission Meets Index Modulation}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Indexes;Indexing;Instant messaging;Modulation;Quantization (signal);Streams;Semantic communication;Training;Ports (computers);PSNR;Semantic communication;joint source-channel coding;digital communication;index modulation;vector quantization},
  doi={10.1109/TCOMM.2026.3720492}}


## Installation

This project is developed based on [BasicSR](https://github.com/xinntao/BasicSR).
So before reproducing the SSC network, please install the BasicSR development environment.

### Installation Steps

Since this project involves modifications to BasicSR, please use the [**Install from a local clone**](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md#install-from-a-local-clone) method.

1. **Install dependent packages**

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Install BasicSR**

   Run the following commands in the project root directory:

   - **Standard Install** (If you don't need C++ extensions):

     \`\`\`bash
     python setup.py develop
     \`\`\`

## Dataset Preparation

Please prepare datasets according to the configuration files in `options/`.
For example, `options/train_SSC_from_scratch.yml` uses DIV2K and Kodak24.

Suggested directory structure:
```text
datasets/
├── DIV2K/
│   ├── DIV2K_train_HR/
│   ├── DIV2K_train_LR_bicubic/
│   └── ...
├── Kodak24/
│   ├── kodim01.png
│   ├── kodim02.png
│   └── ...
```

## Training

The primary training script is `ssc/train.sh`. This script simplifies the process of launching distributed training.

```bash
bash ssc/train.sh
```

**Train SSC from scratch:**

To train the model with different compression ratios, modify the `network_g: args: C` parameter in the configuration file `options/train_SSC_from_scratch.yml` (or your specific config file).

**Using different configurations:**

By default, the script might point to a specific YAML file. You can modify `ssc/train.sh` to use other configurations found in the `options/` directory (e.g., `train_SSC_from_scratch_32C.yml`, etc.).

## Inference

The testing script is `ssc/infer.sh`.

**Benchmarking & Experimental Setup:**

Depending on the experiment you want to run (e.g., comparing FA-IM with QAM, or validating Stream Splitting), you may need to modify the code in `ssc/inference.py` or `ssc/archs/SwinSSC_arch.py`.

1.  **Switching Channel Models (FA-IM vs. FA+QAM):**

    In `ssc/inference.py`, initialize the specific channel class you want to test and pass it to the model.

    *   For **FA-IM** transmission (Proposed):
        ```python
        # ssc/inference.py
        from ssc.channels import FA_IM_Channel
        # ...
        fa_channel_system = FA_IM_Channel(...)
        # ...
        with torch.no_grad():
            output, _, _, _, _ = model.forward_faim(..., channel=fa_channel_system)
        ```

    *   For **Traditional QAM** (Benchmark, w/o IM):
        ```python
        # ssc/inference.py
        from ssc.channels import FA_SISO_Channel # Ensure this class exists/is imported
        # ...
        fa_channel_system = FA_SISO_Channel(...)
        # ...
        with torch.no_grad():
            output, _, _, _, _ = model.forward_faim(..., channel=fa_channel_system)
        ```

2.  **Validating Stream Splitting Scheme:**

    In `ssc/archs/SwinSSC_arch.py`, inside the `forward_faim` method, modify how `noisy_idxs` is calculated to enable or disable the Stream Splitting (SS) mechanism.

    *   **Proposed (Default):**
        ```python
        # ssc/archs/SwinSSC_arch.py
        # noisy_idxs = channel(embed_idxs, chan_param, idx_H) # Default implies ssc=True or controlled by internal logic
        noisy_idxs = channel(embed_idxs, chan_param, idx_H, ssc=True, ssc_adapt=True) # proposed
        ```

    *   **Benchmark (w/o SS):**
        To simulate a baseline without the proposed splitting strategy:
        ```python
        # ssc/archs/SwinSSC_arch.py
        noisy_idxs = channel(embed_idxs, chan_param, idx_H, ssc=False)
        ```

## Acknowledgements

This code is based on [BasicSR](https://github.com/xinntao/BasicSR). We thank the authors for their excellent codebase.
