
"""
This script generates a synthetic atmospheric dataset and saves it as a CSV file.

The dataset includes the following columns:
- `altitude_ft`: Altitude in feet, randomly generated between 28,000 and 40,000 feet.
- `temperature_C`: Temperature in degrees Celsius, randomly generated between -65 and -40.
- `humidity_percent`: Humidity percentage, randomly generated between 10% and 100%.
- `speed_knots`: Speed in knots, randomly generated between 400 and 600 knots.
- `contrail`: A binary column indicating contrail formation (1 for formation, 0 for no formation).
    Contrail formation is determined using a fake formula:
    - Humidity percentage > 60
    - Temperature < -50°C
    - Altitude > 32,000 feet

The generated dataset contains 500 rows and is saved to a file named `fake_contrail_data.csv`.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Fake atmospheric dataset
data_size = 100
data = pd.DataFrame({
    'altitude_ft': np.random.randint(28000, 40000, data_size),
    'temperature_C': np.random.uniform(-65, -40, data_size),
    'humidity_percent': np.random.uniform(10, 100, data_size),
    'speed_knots': np.random.randint(400, 600, data_size),
})

# Contrail formation logic (fake formula)
data['contrail'] = ((data['humidity_percent'] > 60) & 
                    (data['temperature_C'] < -50) & 
                    (data['altitude_ft'] > 32000)).astype(int)

data.to_csv("data/fake_contrail_data.csv", index=False)
