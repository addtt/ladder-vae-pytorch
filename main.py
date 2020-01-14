import boilr
import warnings
from boilr import Trainer

from experiment import LVAEExperiment

BOILR_VERSION = (0, 3, 1)
if boilr.__version_info__[:2] != BOILR_VERSION[:2]:
    msg = "This was last tested with version {}, but the current version is {}"
    msg = msg.format(BOILR_VERSION, boilr.__version_info__)
    warnings.warn(msg)

def main():
    experiment = LVAEExperiment()
    trainer = Trainer(experiment)
    trainer.run()

if __name__ == "__main__":
    main()
