# Singing Voice Synthesis based on Diffusion Model

## Goal
In this project, we design an lightweight singing voice synthesis model while maintaining the synthesis quality of the diffusion model.

## Data
The dataset used for this project can be found in the following directory:
* Dataset folder: `/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/data`

## Components
* **U-net**: Used for processing audio files.
* **mlp-singer**: Deprecated.
* **Results**: Output estimates are stored in `/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/outputs/estimates`.

## Workflow
The `csd datasetloader` combines background music with voice in a nursery rhyme style, and `seperate` is used to isolate the voice.

## Implementation
The `diffusion` model is applied using the CSD dataset for voice synthesis.

## Inference
For detailed implementation and results, refer to the notebook:
* Inference notebook: `/userHome/userhome2/dahyun/voice/Singing_Voice_Synthesis/U-net/test_and_separate.ipynb`

## Members
* [Sang-Hyeong Jin](https://github.com/feVeRin)
* [So-Jeong Kim](https://github.com/ssoojeong)
* [Da-Hyun Song](https://github.com/dahyunnss) 
