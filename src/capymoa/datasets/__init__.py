"""CapyMOA comes with some datasets 'out of the box'. Simply import the dataset
and start using it, the data will be downloaded automatically if it is not
already present in the download directory. You can configure where the datasets
are downloaded to by setting an environment variable (See :mod:`capymoa.env`)

>>> from capymoa.datasets import ElectricityTiny
>>> stream = ElectricityTiny()
>>> stream.next_instance().x
array([0.      , 0.056443, 0.439155, 0.003467, 0.422915, 0.414912])

"""

from ._datasets import (
    KDD99,
    Airlines,
    Bike,
    CovtFD,
    Covtype,
    CovtypeNorm,
    CovtypeTiny,
    Electricity,
    ElectricityTiny,
    Fried,
    FriedTiny,
    Hyper100k,
    Nomao,
    PokerHand,
    RBFm_100k,
    RTG_2abrupt,
    Sensor,
    Spambase,
)
from ._openml import load_openml_dataset
from ._utils import download_unpacked, get_download_dir

__all__ = [
    "KDD99",
    "Airlines",
    "Bike",
    "CovtFD",
    "Covtype",
    "CovtypeNorm",
    "CovtypeTiny",
    "Electricity",
    "ElectricityTiny",
    "Fried",
    "FriedTiny",
    "Hyper100k",
    "Nomao",
    "PokerHand",
    "RBFm_100k",
    "RTG_2abrupt",
    "Sensor",
    "Spambase",
    "download_unpacked",
    "get_download_dir",
    "load_openml_dataset",
]
