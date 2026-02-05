# Just shortcuts
pipeline:
    ./dw-pipeline.py pipeline

status:
    ./dw-pipeline.py status

embed-all:
    ./dw-pipeline.py embed-all

retrieve-all:
    ./dw-pipeline.py retrieve-all

aggregate:
    ./dw-pipeline.py aggregate

clean:
    rm -rf artifacts/retrieval artifacts/weights

clean-all:
    rm -rf artifacts