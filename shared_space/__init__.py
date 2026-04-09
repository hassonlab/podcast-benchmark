# shared_space — VAE denoising plugin for podcast-benchmark.
#
# This __init__.py is intentionally empty to avoid import side-effects
# when train_vae.py imports from shared_space.models.patient_vae.
#
# Registration of VAE components into the podcast-benchmark registry is
# done explicitly via:
#   import shared_space.vae_pipeline
# which is called from main.py only.
