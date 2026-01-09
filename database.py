from tinydb import TinyDB, Query
from models import Plant, Fertilizer, NutrientProfile

db = TinyDB('data/db.json')
plants_table = db.table('plants')
ferts_table = db.table('fertilizers')

def get_all_plants():
    return [Plant(**item) for item in plants_table.all()]

def get_all_fertilizers():
    return [Fertilizer(**item) for item in ferts_table.all()]

def save_plant(plant_data):
    plants_table.insert(plant_data.dict())

def save_fertilizer(fert_data):
    ferts_table.insert(fert_data.dict())