import time
start_time = time.time()

print("Collecting data...")
import create_data
print("Averaging data...")
import create_averages
print("Making customized data...")
import create_training_data
print("Data updated\n")

print("Training new model...")
import make_rnn

print("Predicting outcomes for every match...")
import create_all_predictions

print(f"Finished in {round(time.time() - start_time, 0)} seconds")