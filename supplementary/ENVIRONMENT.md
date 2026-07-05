# Environment Snapshot

- Operating system: Windows
- Project root used for assembly: `D:\sensors`
- Cleaned source folder: `D:\sensors\hra_sl_code_source`
- Python packages required by the public source: see `requirements.txt`
- Main experiment run metadata: `configs/source_diagnosis_run_info.json`
- Shortlisting run metadata: `configs/channel_shortlisting_run_info.json`

The source-diagnosis run metadata records PyTorch `2.7.1+cu118` and CUDA availability on the assembly machine. The code can run on CPU, although tuned autoencoder baselines are slower without GPU.
