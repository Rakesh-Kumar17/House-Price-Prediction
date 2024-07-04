import pickle
import os
import re

# Define a custom unpickler to handle the module name change
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == 'sklearn.linear_model.base':
            renamed_module = 'sklearn.linear_model._base'
        return super().find_class(renamed_module, name)

# Get the current working directory
current_dir = os.path.dirname(__file__)
print("Current working directory:", current_dir)

# Path to the existing model
old_model_path = os.path.join(current_dir, 'artifacts/banglore_home_prices_model.pickle')

# Check if the file exists
if not os.path.exists(old_model_path):
    raise FileNotFoundError(f"File not found: {old_model_path}")

print(f"Loading model from: {old_model_path}")

# Load the old model with the custom unpickler
with open(old_model_path, 'rb') as f:
    model = CustomUnpickler(f).load()

# Path to save the updated model
new_model_path = os.path.join(current_dir, 'artifacts/updated_banglore_home_prices_model.pickle')

# Save the model again using the updated import paths
with open(new_model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model has been updated and saved to {new_model_path}")
