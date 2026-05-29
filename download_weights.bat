@echo off
setlocal

:: Set the checkpoints directory
set CheckpointsDir=models

:: Create necessary directories
mkdir %CheckpointsDir%\musetalk
mkdir %CheckpointsDir%\musetalkV15
mkdir %CheckpointsDir%\syncnet
mkdir %CheckpointsDir%\dwpose
mkdir %CheckpointsDir%\face-parse-bisent
mkdir %CheckpointsDir%\sd-vae-ft-mse
mkdir %CheckpointsDir%\whisper

:: Skip pip -U: huggingface_hub 0.30.2 already installed with huggingface-cli; upgrading drops requirements pin
:: pip install -U "huggingface_hub[hf_xet]"

:: Set HuggingFace endpoint (real HF, not mirror)
set HF_ENDPOINT=https://huggingface.co

:: Download MuseTalk weights
huggingface-cli download TMElyralab/MuseTalk --local-dir %CheckpointsDir%

:: Download SD VAE weights
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir %CheckpointsDir%\sd-vae --include "config.json" "diffusion_pytorch_model.bin"

:: Download Whisper weights
huggingface-cli download openai/whisper-tiny --local-dir %CheckpointsDir%\whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

:: Download DWPose weights
huggingface-cli download yzd-v/DWPose --local-dir %CheckpointsDir%\dwpose --include "dw-ll_ucoco_384.pth"

:: Download SyncNet weights
huggingface-cli download ByteDance/LatentSync --local-dir %CheckpointsDir%\syncnet --include "latentsync_syncnet.pt"

:: Download face-parse-bisent weights
huggingface-cli download ManyOtherFunctions/face-parse-bisent --local-dir %CheckpointsDir%\face-parse-bisent --include "79999_iter.pth" "resnet18-5c106cde.pth"

echo All weights have been downloaded successfully!
endlocal 
