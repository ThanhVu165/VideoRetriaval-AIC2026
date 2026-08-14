# Dataset layout

The competition data stays outside Git. Extract the downloaded archives into the locations configured in `configs/example.yaml`.

Expected logical components:

```text
<data-root>/
├── keyframes/
│   ├── L21_V001/
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   └── ...
│   └── ...
├── clip/
│   └── *.npy / *.npz
├── mapping/
│   └── supplied mapping files
├── media_info/
│   └── supplied media-info files
└── objects/
    └── supplied object JSON files (optional in Phase 0)
```

The audit script deliberately reports the actual mapping/metadata schemas instead of assuming a format. This is important because the competition material states that keyframe filenames are ordered while the original video frame index is supplied in metadata.

Do not commit extracted data, archives, embeddings, or generated artifacts to Git.
