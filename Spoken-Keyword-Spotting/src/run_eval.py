"""Driver: Stages 2-3 only, reusing the CNN saved by Stage 1.

Tunes + fits the One-Class SVM on the saved CNN's embeddings, then evaluates on
the test set. Skips CNN retraining (marvin_kws.h5 already exists).
"""
from model_train import marvin_kws_model
from model_test import marvin_model_test


def main():
    print("========== STAGE 2/3: Tune + fit One-Class SVM ==========", flush=True)
    marvin_kws_model()

    print("========== STAGE 3/3: Evaluate on test set ==========", flush=True)
    marvin_model_test()

    print("========== DONE ==========", flush=True)


if __name__ == "__main__":
    main()
