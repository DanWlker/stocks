# Build the page from the data already in data/
default: build

# Generate site/index.html and the replay files in site/anim/
build:
    uv run python export_animation.py

# Re-download the index data from Yahoo Finance into data/ (results can shift if Yahoo revises its history)
fetch:
    uv run python fetch_data.py

# Build, then serve site/ locally: the replays load over HTTP, so a double-clicked index.html can't show them
serve port="8000": build
    uv run python -m http.server {{port}} --bind 127.0.0.1 --directory site
