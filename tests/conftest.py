import os

os.environ.setdefault("MPLBACKEND", "Agg")  # must be set before importing pyplot

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)  # guarantee no GUI backend is used
