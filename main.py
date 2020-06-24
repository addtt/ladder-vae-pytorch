from boilr import Trainer

from experiment import LVAEExperiment


def main():
    experiment = LVAEExperiment()
    trainer = Trainer(experiment)
    trainer.run()


if __name__ == "__main__":
    main()
