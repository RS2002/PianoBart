
# EMOPIA

- `midis/`: midi files transcribed using GiantMIDI.
    * Filename `Q1_xxxxxxx_2.mp3`: Q1 means this clips belongs to Q1 on V-A space; xxxxxxx is the song ID on YouTube; and the `2` means this clip is the 2nd clips taken from the full song.
- `metadata/`: metadata from YouTube. (Got when crawling)
- `songs_lists/`: YouTube URLs of songs.
- `tagging_lists/`: raw tagging result for each sample.
- `label.csv`: metadata that record filename, clip timestamps, and annotator.
- `metadata_by_song.csv`: list all the clips by the song. Can be used to create the train/val/test splits to avoid the same song appear in both train and test.
- `scripts/prepare_split.ipynb`: the script to create train/val/test splits and save them to csv files.
