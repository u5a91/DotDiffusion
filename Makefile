.PHONY: train clean

train:
	uv run python -m scripts.train --config configs/base.yaml

clean:
	rm -rf outputs checkpoints
	mkdir -p outputs checkpoints
