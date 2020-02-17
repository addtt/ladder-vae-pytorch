import warnings

import boilr
import multiobject
from boilr import Trainer

from experiment import LVAEExperiment


def _check_version(pkg, pkg_str, version):
    if pkg.__version_info__[:2] != version[:2]:
        msg = "This was last tested with {} version {}, but the current version is {}"
        msg = msg.format(pkg_str, version, pkg.__version_info__)
        warnings.warn(msg)

BOILR_VERSION = (0, 6, 0)
MULTIOBJ_VERSION = (0, 0, 3)
_check_version(boilr, 'boilr', BOILR_VERSION)
_check_version(multiobject, 'multiobject', MULTIOBJ_VERSION)

def main():
    experiment = LVAEExperiment()
    trainer = Trainer(experiment)
    trainer.run()

if __name__ == "__main__":
    main()
