# main.py
import config
from src.data_loader import load_and_process_data
from src.modeling import train_and_evaluate

def main():
    """
    Main execution pipeline for the mutual fund performance prediction project.
    """
    # Steps 1-3: Load, process, and engineer features.
    processed_df = load_and_process_data(config.RAW_DATA_PATH)

    # Step 4: Train the model and evaluate its performance.
    train_and_evaluate(processed_df)

if __name__ == "__main__":
    main()