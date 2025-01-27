import click

from ml.utils import (
    train_status_classification_macross_model,
    train_attack_classification_macross_model,
    train_binary_classification_macross_model,
    train_scenario_classification_macross_model,
)


@click.command()
@click.option(
    "-d",
    "--data_path",
    help="training data dir path containing parquet files",
    required=True,
)
@click.option("-m", "--model_path", help="output model path", required=True)
@click.option(
    "-t",
    "--task",
    help='classification task. Option: "status" or "attack" or "binary" or "scenario"',
    required=True,
)
def main(data_path, model_path, task):
    if task == "status":
        train_status_classification_macross_model(data_path, model_path)
    elif task == "attack":
        train_attack_classification_macross_model(data_path, model_path)
    elif task == "binary":
        train_binary_classification_macross_model(data_path, model_path)
    elif task == "scenario":
        train_scenario_classification_macross_model(data_path, model_path)
    else:
        exit("Not Support")


if __name__ == "__main__":
    main()
