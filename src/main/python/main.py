from src.main.python.package.runners.runner import Runner

runner = Runner(runner_hyperparameters_name="SETimesByT5Vaswani2017Kocmi2018_0")

runner.load_dataset()
runner.load_model()
runner.load_trainer()
runner.run_trainer()

