#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os
import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    # Initialize W&B run
    run = wandb.init(
        project="nyc_airbnb", 
        job_type="basic_cleaning", 
        group="cleaning", 
        save_code=True
    )
    run.config.update(args)

    # Download the input artifact
    artifact = run.use_artifact(args.input_artifact)
    artifact_dir = artifact.download()  # Downloads all files to a folder

    # Dynamically find the CSV file in the artifact folder
    csv_files = glob.glob(os.path.join(artifact_dir, "*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found in {artifact_dir}")
    input_csv_path = csv_files[0]  # use the first CSV found

    # Read the CSV
    df = pd.read_csv(input_csv_path)

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    # Keep only rows within valid coordinates
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned file
    cleaned_filename = "clean_sample.csv"
    df.to_csv(cleaned_filename, index=False)

    # Log the cleaned dataset as a new artifact
    cleaned_artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    cleaned_artifact.add_file(cleaned_filename)
    run.log_artifact(cleaned_artifact)

    logger.info(f"Cleaned data saved and logged as artifact: {args.output_artifact}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact in W&B to be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output cleaned data artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact, e.g., 'clean_data'",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the cleaned dataset",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to include in the dataset (filter outliers)",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to include in the dataset (filter outliers)",
        required=True
    )

    args = parser.parse_args()
    go(args)
