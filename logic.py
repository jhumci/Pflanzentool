import numpy as np
from scipy.optimize import minimize

# Umrechnungsfaktoren (Oxid -> Elementar)
CONVERSION = {
    "P2O5_to_P": 0.4364,
    "K2O_to_K": 0.8302,
    "MgO_to_Mg": 0.6032,
    "CaO_to_Ca": 0.7147,
    "SO3_to_S": 0.4005
}

def optimize_nutrients(target_profile, fertilizers):
    """
    Findet die beste Mischung der Dünger (ml/L), um das Zielprofil zu erreichen.
    """
    nutrient_keys = ['n', 'p', 'k', 'ca', 'mg', 's']
    target_vector = np.array([getattr(target_profile, k) for k in nutrient_keys])
    
    # Matrix der Düngerzusammensetzungen
    fert_matrix = []
    for f in fertilizers:
        fert_matrix.append([getattr(f.composition, k) for k in nutrient_keys])
    fert_matrix = np.array(fert_matrix).T

    # Zielfunktion: Summe der quadratischen Abweichungen minimieren
    def objective(amounts):
        current_profile = fert_matrix @ amounts
        return np.sum((current_profile - target_vector)**2)

    # Startwerte (alle 0 ml) und Constraints (keine negativen Mengen)
    initial_guess = np.zeros(len(fertilizers))
    bounds = [(0, 10) for _ in range(len(fertilizers))] # Max 10ml/L als Annahme

    res = minimize(objective, initial_guess, bounds=bounds)
    return res.x # Gibt Liste der ml pro Dünger zurück