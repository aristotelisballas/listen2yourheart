from challengemodel import my_load_challenge_model
from challengerun import my_run_challenge_model
from challengetrain import my_train_challenge_model


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    my_train_challenge_model(data_folder, model_folder, verbose)


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    return my_load_challenge_model(model_folder, verbose)


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    return my_run_challenge_model(model, data, recordings, verbose)


# Save your trained model.
def save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier):
    # NOTE: disabling this function since models are saved progressively during training
    pass


# Extract features from the data.
def get_features(data, recordings):
    pass
