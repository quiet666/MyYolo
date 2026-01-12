from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,  # Use FP16 for faster inference
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image("/root/autodl-tmp/ultralytics-main/datas/mirror/G01-1.png")

# Query with multiple text prompts
results = predictor(text=["Gold wire reflection", "dirt"])

# Works with descriptive phrases
# results = predictor(text=["person with red cloth", "person with blue cloth"])

# Query with a single concept
# results = predictor(text=["a person"])