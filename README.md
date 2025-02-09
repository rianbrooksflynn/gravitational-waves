# gravitational-waves
Rian Brooks Flynn's code for the [NSF HDR A3D3 Gravitational Waves ML Challenge](https://www.codabench.org/competitions/2626/).

## Model Structure
The model is a dual-stream transformer model that uses a transformer stream to encode and then decode the time series data and a CNN stream to encode and decode the frequency domain data.

The transformer stream is a standard transformer encoder with a sinusoidal positional encoding and standard multi-head attention with GELU activations and layer normalization. The output of the transformer stream is passed through a feed-forward network to produce a reconstruction of the time series data.

The time-series data passes through a short-time Fourier transform (STFT) to produce a frequency domain representation. This frequency domain representation is passed through a CNN with a ReLU activation and max pooling. The output of the CNN stream is passed through a feed-forward network to produce a reconstruction of the frequency domain data. The frequency domain data is then passed through an inverse short-time Fourier transform (ISTFT) to produce a reconstruction of the time-series data.

The two reconstructions are concatenated and passed through a final feed-forward network to produce a single final reconstruction.

## Training
The model is trained using a combination of reconstruction loss, KL divergence loss, and contrastive loss (for the encoded features, between the different types of signals provided in the training data). The anomaly scores are computed using KL divergence.

## Team Members
Rian Brooks Flynn (rbflynn@purdue.edu) - Purdue University

## Running the Code
1. Clone the repository.
2. Download the datasets from the [Gravitational Waves ML Challenge](https://www.codabench.org/competitions/2626/), unzip them, and place them in the `datasets` directory.
3. Create a fresh conda environment and install the requirements file:
```
conda create -n gravitational-waves python=3.11
conda activate gravitational-waves
pip install -r training_requirements.txt
```
4. Run the training script with `python3 training/train_final_model.py`.
5. Run the test script with `python3 training/test_final_model.py`.

## References
See `references.bib` for a list of references.

## Citation
If you use this code, please cite it as shown in the GitHub sidebar.
