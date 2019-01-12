# Setup

Before starting to develop, add the project's root directory to PYTHONPATH on your system, so that module imports are resolved correctly.

We recommend you create a virtual environment (like conda) and activate it. You can then install the dependencies via
`pip install -r requirements.txt` or `conda install --yes --file requirements.txt`

For running on the GPU, you will need to install the tensorflow-gpu package rather than tensorflow. The rest of the requirements ought to be the same, but you can install them via requirements-gpu.txt.

