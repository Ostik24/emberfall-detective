# Running the notebook in Colab

The notebook needs the repository plus three assets that are not in git: the extended catalog
(`data/catalog_extended.parquet`), the fine-tuned retriever (`models/E3/`) and the caches
(`.cache/`: Claude, TMDB and document-vector results). They travel as one zip (about 550 MB) on
Google Drive, shared by link:
https://drive.google.com/file/d/1OsDhPC5UjxfKSLCpwVF5rPZsh3sjGyA8/view

With the caches, every model call in the notebook replays from disk: no API key, no cost, the
same outputs. The zip never contains `.env`.

## Steps

1. Open `notebooks/Emberfall_Detective.ipynb` in Colab (File -> Open notebook -> GitHub, or
   upload it). Keep the default CPU runtime: GPU inference runs in fp16, which changes rankings
   slightly and then misses the caches.
2. Runtime -> Run all. The first cell downloads the zip from the link above, unzips it to
   `/content/emberfall_detective` and installs the package (a few minutes).
3. The first time, the cell stops with "Restart session, then Run all again": the install
   replaced libraries Colab had already loaded (numpy / pandas / torch). Do exactly that
   (Runtime -> Restart session, not "delete runtime"); the second run takes seconds.

Only the Gradio demo and questions the cache has not seen need keys: add
`EMBERFALL_CLAUDE_KEY` and `TMDB_API_KEY` in Colab Secrets (the key icon on the left), never
in a cell.

## Rebuilding the zip

`uv run python data/build_colab_bundle.py` writes it next to the repository folder (git-tracked
files + the three assets). Replace the Drive file with a new version (Drive: right click ->
File information -> Manage versions -> Upload new version) so the shared link stays the same.
