# Bark

```bash
# Download weights
huggingface-cli download suno/bark coarse.pt fine.pt text.pt

# Convert to npz format
python convert.py weights/

# Run the model
python model.py weights/
```