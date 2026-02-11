# Just shortcuts

# Run full pipeline; optional dataset argument restricts to a single dataset.
pipeline dataset='':
    if [ "{{dataset}}" = "" ]; then \
      ./dw-pipeline.py pipeline; \
    else \
      ./dw-pipeline.py pipeline {{dataset}}; \
    fi

status:
    ./dw-pipeline.py status

# Embed embeddings; optional dataset argument restricts to a single dataset.
embed dataset='':
    if [ "{{dataset}}" = "" ]; then \
      ./dw-pipeline.py embed; \
    else \
      ./dw-pipeline.py embed {{dataset}}; \
    fi

# 1-NN retrieval; optional dataset argument restricts to a single candidate.
retrieve dataset='':
    if [ "{{dataset}}" = "" ]; then \
      ./dw-pipeline.py retrieve; \
    else \
      ./dw-pipeline.py retrieve {{dataset}}; \
    fi

aggregate:
    ./dw-pipeline.py aggregate

# Clean artifacts. With a dataset argument only that dataset's artifacts are
# removed; without an argument, everything is deleted after a confirmation.
[confirm]
clean dataset='':
    if [ "{{dataset}}" = "" ]; then \
      ./dw-pipeline.py clean; \
    else \
      ./dw-pipeline.py clean {{dataset}}; \
    fi