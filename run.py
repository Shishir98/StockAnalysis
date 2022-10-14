from api import BaseAPI, TestAPI
from constants import TRAIN_TEST_SIZE, LEARNING_RATE, EPOCH, BATCH_SIZE, VALIDATION_SPLIT, LOSS, DATA_DIR, \
    TEST_DATA_DIR, TIME_STEPS


def run():
    api = BaseAPI(DATA_DIR, TIME_STEPS, TRAIN_TEST_SIZE, LEARNING_RATE, EPOCH, BATCH_SIZE, VALIDATION_SPLIT, LOSS)
    model, history = api.train_base_model()
    test_api = TestAPI(TEST_DATA_DIR, TIME_STEPS, TRAIN_TEST_SIZE, LEARNING_RATE, EPOCH, BATCH_SIZE, VALIDATION_SPLIT,
                       LOSS,
                       model)
    model, history = api.train_base_model()


if __name__ == "__main__":
    run()
else:
    print("STOP")
