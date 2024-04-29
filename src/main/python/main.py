
from package.runners.runner_setimesbyt5_vaswani2017_kocmi2018 import RunnerSetimesbyt5Vaswani2017Kocmi2018

runner = RunnerSetimesbyt5Vaswani2017Kocmi2018()

runner.load_dataset()
runner.load_model()
runner.run_trainer()


