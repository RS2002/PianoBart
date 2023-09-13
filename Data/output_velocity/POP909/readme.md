# Velocity Classification

Dataset: POP909

classify Pitch events into six classes with the POP909 dataset. MIDI velocity values range from 0–127, and we quantize the information into six categories, pp (0–31), p (32–47), mp (48–63), mf (64–79), f (80–95), and ff (96–127)

   ```json
    {
        "pp": 0,
        "p": 1,
        "mp": 2,
        "mf": 3,
        "f": 4,
        "ff": 5,
        "OTHER": 6 // includes EOS/PAD
    }
   ```
