from .rustevo import RustEvoDataset
from .pyevo import PyEvoDataset

DS_DICT = {
    "rustevo": RustEvoDataset,
    "pyevo": PyEvoDataset,
}
