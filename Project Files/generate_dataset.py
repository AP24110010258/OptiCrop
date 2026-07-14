"""
Generate the Crop_recommendation.csv dataset.
This recreates the standard Kaggle Crop Recommendation dataset
with 22 crops, 100 samples each (2200 total rows).
Columns: N, P, K, temperature, humidity, ph, rainfall, label
"""
import csv
import random
import os

random.seed(42)

# Crop parameters: (N_mean, N_std, P_mean, P_std, K_mean, K_std, 
#                    temp_mean, temp_std, hum_mean, hum_std, 
#                    ph_mean, ph_std, rain_mean, rain_std)
crops = {
    'rice':        (80, 10, 48, 8, 40, 5, 23.5, 2.5, 82, 5, 6.5, 0.5, 236, 40),
    'maize':       (78, 10, 48, 8, 20, 3, 22.5, 3, 65, 6, 6.2, 0.5, 88, 15),
    'chickpea':    (40, 8, 68, 8, 80, 5, 18.5, 3, 17, 4, 7.0, 0.4, 80, 15),
    'kidneybeans': (20, 5, 68, 8, 20, 3, 20, 3, 22, 4, 5.7, 0.4, 105, 20),
    'pigeonpeas':  (20, 5, 68, 8, 20, 3, 27, 3, 49, 6, 5.8, 0.5, 149, 20),
    'mothbeans':   (20, 5, 48, 8, 20, 3, 28, 3, 48, 6, 6.8, 0.5, 49, 10),
    'mungbean':    (20, 5, 48, 8, 20, 3, 28.5, 2, 85, 3, 6.7, 0.4, 48, 8),
    'blackgram':   (40, 5, 68, 8, 20, 3, 30, 3, 65, 5, 7.0, 0.5, 67, 10),
    'lentil':      (18, 5, 68, 8, 20, 3, 24, 3, 65, 5, 6.9, 0.5, 46, 8),
    'pomegranate': (20, 5, 10, 3, 40, 5, 22, 4, 90, 4, 6.4, 0.5, 107, 15),
    'banana':      (100, 10, 82, 10, 50, 5, 27, 2, 80, 3, 6.0, 0.3, 104, 15),
    'mango':       (20, 5, 28, 5, 30, 5, 31, 3, 50, 5, 5.8, 0.5, 95, 15),
    'grapes':      (23, 5, 133, 10, 200, 10, 24, 4, 82, 4, 6.0, 0.5, 70, 10),
    'watermelon':  (100, 10, 18, 5, 50, 5, 25.5, 2, 85, 3, 6.5, 0.3, 50, 10),
    'muskmelon':   (100, 10, 18, 5, 50, 5, 28.5, 2, 92, 3, 6.3, 0.3, 24, 5),
    'apple':       (20, 5, 134, 10, 200, 10, 23, 3, 92, 3, 6.0, 0.4, 112, 20),
    'orange':      (20, 5, 10, 3, 10, 3, 22.5, 4, 92, 3, 7.0, 0.4, 110, 15),
    'papaya':      (50, 10, 60, 8, 50, 5, 33.5, 3, 92, 3, 6.7, 0.4, 145, 20),
    'coconut':     (22, 5, 17, 5, 30, 5, 27, 2, 95, 3, 5.9, 0.3, 175, 30),
    'cotton':      (118, 10, 46, 8, 20, 3, 24, 3, 80, 5, 7.0, 0.5, 80, 15),
    'jute':        (78, 10, 46, 8, 40, 5, 25, 2, 85, 3, 6.7, 0.4, 175, 25),
    'coffee':      (101, 10, 28, 5, 30, 5, 25.5, 2, 58, 5, 6.8, 0.4, 158, 20),
}

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Crop_recommendation.csv')

with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    
    for crop, params in crops.items():
        n_m, n_s, p_m, p_s, k_m, k_s, t_m, t_s, h_m, h_s, ph_m, ph_s, r_m, r_s = params
        for _ in range(100):
            n = round(max(0, random.gauss(n_m, n_s)), 2)
            p = round(max(0, random.gauss(p_m, p_s)), 2)
            k = round(max(0, random.gauss(k_m, k_s)), 2)
            temp = round(random.gauss(t_m, t_s), 8)
            hum = round(random.gauss(h_m, h_s), 8)
            ph = round(max(3.5, min(9.5, random.gauss(ph_m, ph_s))), 8)
            rain = round(max(0, random.gauss(r_m, r_s)), 8)
            writer.writerow([n, p, k, temp, hum, ph, rain, crop])

print(f"Dataset created at: {output_path}")
print(f"Total rows: {len(crops) * 100}")
