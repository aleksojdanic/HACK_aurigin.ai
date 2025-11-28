



from hf_trainer import train_random_forest, create_submission_csv


def main():
    print("Aurigin.ai hackathon – HF + RandomForest")

    model, X_test, test_split = train_random_forest()

    print("Creating submission CSV...")
    create_submission_csv(model, X_test, test_split, out_path="submission.csv")

    print("Done.")


if __name__ == "__main__":
    main()

