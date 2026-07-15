"""Driver: full retrain + evaluate in one pass.

Runs the CNN feature-extractor training, the One-Class SVM tuning/fit, and the
test-set evaluation sequentially. The dataset is downloaded on the first call
and reused thereafter.
"""
from model_train import model_train, marvin_kws_model
from model_test import marvin_model_test


def main():
    print("========== STAGE 1/3: Train CNN feature extractor ==========", flush=True)
    model_train()

    print("========== STAGE 2/3: Tune + fit One-Class SVM ==========", flush=True)
    marvin_kws_model()

    print("========== STAGE 3/3: Evaluate on test set ==========", flush=True)
    marvin_model_test()

    print("========== DONE ==========", flush=True)


if __name__ == "__main__":
    main()
