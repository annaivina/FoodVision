from src.data_loader import DataLoader
from src.model import ModelBuilder
from src.trainer import Trainer
from src.callbacks import get_callbacks
from src.load_config import load_config
import argparse



def run_experiment(mode="", experiment_name="example_experiment", config_path = "test.yaml", model_name="effb0"):

    config = load_config(config_path)

    #Loading the data
    data_loader = DataLoader(dataset_name=config['dataset_name'], batch_size=config['batch_size'])
    train_data, test_data = data_loader.load_data()
    num_classes = data_loader.num_classes


    #Initiliase the model
    model_build = ModelBuilder(input_shape=config["input_shape"], num_classes=num_classes, activation=config['activation'], lr=config["learning_rate"], mixed_precision_training=False)

    if mode == 'train':
        print("\n🟢 Starting Model Training...")
        callbacks = get_callbacks(experiment_name=experiment_name, model_name=model_name, is_fine_tune=config['fine_tune'])

        trainer = Trainer(model_build.model, train_data, test_data, epochs=config['epochs'], callbacks=callbacks)
        trainer.train()
        model_build.save_model(f"{experiment_name}/models/{model_name}.keras")

        if config['fine_tune']:
            print("\n🟢 Loading the model for fine-tuning")
            model_build.load_model(f"models/{model_name}.keras")
            print("\n🟢 Fine tuning the model...")
            model_build.fine_tune(trainable_layers=config['fine_tune_layers'], learning_rate=config['fine_tune_lr'])
            trainer.train()
            
            model_build.save_model(f"{experiment_name}/models/{model_name}_fine_tuned.keras")


    elif mode == "evaluate":
        print("\n🔵 Evaluating the Model...")
        model_build.load_model(f"{experiment_name}/models/{model_name}.keras")
        trainer = Trainer(model_build.model, train_data, test_data)
        trainer.evaluate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the Food101 EfficientNet model.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"], help="Mode: train or evaluate")
    parser.add_argument("--experiment_name", type=str, default="food101_experiment", help="Experiment name for logging.")
    parser.add_argument("--config_path", type=str, default="configs/test.yaml", help="Provide the config name")
    parser.add_argument("--model_name", type=str, default="efficientnetb0", help="Model filename to save/load.")

    
    args = parser.parse_args()
    run_experiment(mode=args.mode, experiment_name=args.experiment_name, config_path=args.config_path, model_name=args.model_name)