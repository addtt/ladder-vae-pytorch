from experiment.config import ExperimentConfig
from framework.trainer import Trainer


def main():
    experiment = ExperimentConfig()
    trainer = Trainer(experiment)
    trainer.run()

if __name__ == "__main__":
    main()
