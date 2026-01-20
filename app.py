import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import altair as alt
from nutrient_profile import NutrientProfile
from tinydb import TinyDB, Query
from scipy.optimize import minimize
from datetime import datetime

# --- KONFIGURATION & DB ---
st.set_page_config(page_title="Hydro Optimizer", layout="wide")
db = TinyDB('db.json')
plants_table = db.table('plants')
ferts_table = db.table('fertilizers')
logs_table = db.table('logs')

# Wenn eine alternative JSON-DB vorhanden ist (z.B. data/db.json),
# beim Start in die TinyDB-Tabellen importieren (ohne Duplikate).
external_db_path = os.path.join('data', 'db.json')
if os.path.exists(external_db_path):
    try:
        with open(external_db_path, 'r', encoding='utf-8') as f:
            ext = json.load(f)

        # Wasserqualität aus externer Datei übernehmen, falls vorhanden
        if isinstance(ext, dict) and ext.get('water_source'):
            first_ws = next(iter(ext['water_source'].values()))
            if isinstance(first_ws, dict) and first_ws.get('values'):
                ext_water_quality = first_ws['values']

        # Pflanzen importieren
        for _id, p in (ext.get('plants') or {}).items():
            name = p.get('name')
            phases = p.get('phases', {})
            if name and not plants_table.search(Query().name == name):
                plants_table.insert({'name': name, 'phases': phases})

        # Düngemittel importieren
        for _id, fert in (ext.get('fertilizers') or {}).items():
            name = fert.get('name')
            price = fert.get('price_per_ml', 0)
            composition = fert.get('composition', {})
            if name and not ferts_table.search(Query().name == name):
                ferts_table.insert({'name': name, 'price_per_ml': price, 'composition': composition})
    except Exception as e:
        try:
            st.warning(f"Externe DB nicht geladen: {e}")
        except Exception:
            pass

# Umrechnungsfaktoren
CONV = {
    "P2O5_P": 0.4364,
    "K2O_K": 0.8302,
    "MgO_Mg": 0.6032,
    "CaO_Ca": 0.7147,
    "SO3_S": 0.4005
}

# Standard Wasserqualität (Dagersheim)
WATER_QUALITY = {"n": 1.3, "p": 0.2, "k": 1.6, "ca": 63.2, "mg": 14.1, "s": 17.2}

# Falls von der externen DB Wasserqualitätswerte geladen wurden, überschreiben
if 'ext_water_quality' in globals():
    WATER_QUALITY = ext_water_quality

# --- LOGIK FUNKTIONEN ---
def run_optimization(target_profile, available_ferts, water_profile):
    """Berechnet die optimale Menge ml/L pro Dünger."""
    nutrient_keys = ['n', 'p', 'k', 'ca', 'mg', 's']
    
    # Netto-Bedarf (Ziel - Wasser)
    target_netto = np.array([max(0, target_profile[k] - water_profile[k]) for k in nutrient_keys])
    
    # Matrix der Dünger-Inhaltsstoffe
    fert_names = [f['name'] for f in available_ferts]
    matrix = []
    for f in available_ferts:
        matrix.append([f['composition'].get(k, 0) for k in nutrient_keys])
    matrix = np.array(matrix).T # Transponieren für Matrix-Multiplikation

    # Zielfunktion: Fehlerquadrate minimieren
    def objective(amounts):
        prediction = matrix @ amounts
        return np.sum((prediction - target_netto)**2)

    # Constraints: Keine negativen Mengen, Max 15ml/L
    bounds = [(0, 15) for _ in range(len(available_ferts))]
    res = minimize(objective, np.zeros(len(available_ferts)), bounds=bounds)
    
    return res.x, target_netto, (matrix @ res.x)

# --- UI NAVIGATION ---
st.title("🌿 Hydroponik Nährstoff-Optimizer")
tab1, tab2, tab3, tab4 = st.tabs(["🧮 Optimierung", "🧪 Düngemittel", "🌱 Pflanzen", "📋 Logs"])

# --- TAB 1: OPTIMIERUNG ---
with tab1:
    st.header("Mischung berechnen")
    
    all_plants = plants_table.all()
    all_ferts = ferts_table.all()

    if not all_plants or not all_ferts:
        st.warning("Bitte lege zuerst Pflanzen und Düngemittel in den anderen Tabs an!")
    else:
        col1, col2 = st.columns([2, 2])
        
        with col1:
            st.subheader("Wasserqualität")
            water_type = st.selectbox(
                "Wasser wählen",
                ["Leitungswasser Dagersheim", "Destilliertes Wasser"],
                key="water_select"
            )
            
            if water_type == "Destilliertes Wasser":
                display_water = {"n": 0, "p": 0, "k": 0, "ca": 0, "mg": 0, "s": 0}
            else:
                display_water = WATER_QUALITY
            
            st.write("**Zusammensetzung (mg/L):**")
            water_df = pd.DataFrame({
                "Element": ["N", "P", "K", "Ca", "Mg", "S"],
                "Menge (mg/L)": [display_water["n"], display_water["p"], display_water["k"], 
                                 display_water["ca"], display_water["mg"], display_water["s"]]
            })
            st.table(water_df)
        
        with col2:
            st.subheader("Nährstoffziel")
            target_plant = st.selectbox(
                "Pflanze wählen",
                [p['name'] for p in all_plants],
                key="plant_select"
            )
            plant = next(p for p in all_plants if p['name'] == target_plant)
            
            target_phase = st.selectbox(
                "Wachstumsphase",
                list(plant['phases'].keys()),
                key="phase_select"
            )
            target_vals = plant['phases'][target_phase]
            
            st.write("**Zielwerte (mg/L):**")
            target_df = pd.DataFrame({
                "Element": ["N", "P", "K", "Ca", "Mg", "S"],
                "Ziel (mg/L)": [target_vals["n"], target_vals["p"], target_vals["k"],
                               target_vals["ca"], target_vals["mg"], target_vals["s"]]
            })
            st.table(target_df)
        
        st.divider()
        
        liters = st.number_input("Menge der Nährlösung (Liter)", min_value=1.0, value=10.0, step=1.0)
        
        if st.button("🚀 Optimale Mischung berechnen", type="primary", key="calc_mix_btn"):
            opt_result = run_optimization(target_vals, all_ferts, display_water)
            st.session_state.calc_results = {
                'amounts': opt_result[0],
                'netto_req': opt_result[1],
                'achieved': opt_result[2],
                'water_profile': display_water,
                'plant': target_plant,
                'phase': target_phase,
                'water_type': water_type,
                'liters': liters
            }
        
        # Zeige Ergebnisse, falls vorhanden
        if 'calc_results' in st.session_state:
            res = st.session_state.calc_results
            amounts = res['amounts']
            netto_req = res['netto_req']
            achieved = res['achieved']
            liters = res['liters']
            target_plant = res['plant']
            target_phase = res['phase']
            water_type = res['water_type']
            
            st.subheader(f"Ergebnis für {liters} Liter")
            
            # Tabelle: Was man mischen muss
            res_data = []
            total_cost = 0
            total_fert_ml = 0
            
            for i, fert in enumerate(all_ferts):
                if amounts[i] > 0.01:
                    ml_needed = amounts[i] * liters
                    cost = ml_needed * fert.get('price_per_ml', 0)
                    res_data.append({
                        "Komponente": fert['name'],
                        f"Menge pro {liters}L": f"{round(ml_needed, 2)} ml",
                        "Kosten (€)": f"{cost:.2f} €"
                    })
                    total_cost += cost
                    total_fert_ml += ml_needed
            
            # Wasser als Rest berechnen
            water_ml = (liters * 1000) - total_fert_ml
            res_data.append({
                "Komponente": "Leitungswasser" if water_type == "Leitungswasser Dagersheim" else "Destilliertes Wasser",
                f"Menge pro {liters}L": f"{round(water_ml, 2)} ml",
                "Kosten (€)": "0.00 €"
            })
            
            result_df = pd.DataFrame(res_data)
            st.table(result_df)
            st.metric("Gesamtkosten", f"{total_cost:.2f} €")
            
            st.divider()
            st.subheader("Profil-Abgleich (Ziel vs. Mischung)")

            nutrient_keys = ['n', 'p', 'k', 'ca', 'mg', 's']
            matrix = []
            for f in all_ferts:
                matrix.append([f['composition'].get(k, 0) for k in nutrient_keys])
            matrix = np.array(matrix).T

            fert_names = [f['name'] for f in all_ferts]

            chart = NutrientProfile.layered_mixture_vs_target(netto_req, matrix, amounts, fert_names, res['water_profile'])
            st.altair_chart(chart, use_container_width=True)

            st.divider()
            st.subheader("Volumenanteile der Mischung")
            
            col1, col2 = st.columns(2)
            
            # Mit Wasser
            with col1:
                pie_vol_df_with_water = NutrientProfile.pie_df_volume(amounts, fert_names, liters)
                if not pie_vol_df_with_water.empty:
                    pie_vol_chart = NutrientProfile.pie_chart_volume(pie_vol_df_with_water, f"Mit Wasser ({liters}L)")
                    st.altair_chart(pie_vol_chart, use_container_width=True)
            
            # Ohne Wasser
            with col2:
                pie_vol_df_no_water = NutrientProfile.pie_df_volume_no_water(amounts, fert_names)
                if not pie_vol_df_no_water.empty:
                    pie_vol_chart_no_water = NutrientProfile.pie_chart_volume(pie_vol_df_no_water, "Nur Dünger")
                    st.altair_chart(pie_vol_chart_no_water, use_container_width=True)
            
            st.divider()
            
            # Log-Button
            if st.button("📝 Diese Mischung loggen", type="secondary", key="log_mix_btn"):
                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "plant": target_plant,
                    "phase": target_phase,
                    "water_type": water_type,
                    "liters": liters,
                    "fertilizers": [
                        {
                            "name": fert['name'],
                            "ml": round(amounts[i] * liters, 2)
                        }
                        for i, fert in enumerate(all_ferts)
                        if amounts[i] > 0.01
                    ],
                    "total_cost": round(total_cost, 2)
                }
                logs_table.insert(log_entry)
                st.session_state.log_success = True
            
            if st.session_state.get('log_success', False):
                st.success("✅ Mischung gespeichert in der Datenbank!")

# --- TAB 2: DÜNGEMITTEL ---
with tab2:
    st.header("Düngemittel-Datenbank")
    
    # Build table data from database
    fert_table_data = []
    for f in ferts_table.all():
        comp = f.get('composition', {})
        # Reverse-convert elemental mg/ml to oxide percentages
        n_pct = comp.get('n', 0.0) / 10.0
        p2o5_pct = comp.get('p', 0.0) / (10.0 * CONV['P2O5_P']) if CONV['P2O5_P'] else 0.0
        k2o_pct = comp.get('k', 0.0) / (10.0 * CONV['K2O_K']) if CONV['K2O_K'] else 0.0
        cao_pct = comp.get('ca', 0.0) / (10.0 * CONV['CaO_Ca']) if CONV['CaO_Ca'] else 0.0
        mgo_pct = comp.get('mg', 0.0) / (10.0 * CONV['MgO_Mg']) if CONV['MgO_Mg'] else 0.0
        so3_pct = comp.get('s', 0.0) / (10.0 * CONV['SO3_S']) if CONV['SO3_S'] else 0.0
        
        fert_table_data.append({
            "Name": f['name'],
            "Preis €/ml": f.get('price_per_ml', 0.0),
            "N %": round(n_pct, 3),
            "P2O5 %": round(p2o5_pct, 3),
            "K2O %": round(k2o_pct, 3),
            "CaO %": round(cao_pct, 3),
            "MgO %": round(mgo_pct, 3),
            "SO3 %": round(so3_pct, 3)
        })
    
    df_ferts = pd.DataFrame(fert_table_data)
    
    st.subheader("Dünger bearbeiten")
    st.info("Bearbeite die Tabelle direkt. Neue Zeilen werden automatisch hinzugefügt.")
    
    edited_ferts = st.data_editor(
        df_ferts,
        use_container_width=True,
        num_rows="dynamic",
        key="ferts_editor"
    )
    
    # Save changes back to database
    if st.button("💾 Änderungen speichern", type="primary", key="save_ferts_btn"):
        # Clear old data
        ferts_table.truncate()
        
        # Rebuild from edited dataframe
        for _, row in edited_ferts.iterrows():
            name = row['Name']
            if name:  # Skip empty rows
                # Convert oxide percentages back to elemental mg/ml
                new_fert = {
                    "name": name,
                    "price_per_ml": float(row['Preis €/ml']) if pd.notna(row['Preis €/ml']) else 0.0,
                    "composition": {
                        "n": float(row['N %']) * 10 if pd.notna(row['N %']) else 0.0,
                        "p": float(row['P2O5 %']) * 10 * CONV["P2O5_P"] if pd.notna(row['P2O5 %']) else 0.0,
                        "k": float(row['K2O %']) * 10 * CONV["K2O_K"] if pd.notna(row['K2O %']) else 0.0,
                        "ca": float(row['CaO %']) * 10 * CONV["CaO_Ca"] if pd.notna(row['CaO %']) else 0.0,
                        "mg": float(row['MgO %']) * 10 * CONV["MgO_Mg"] if pd.notna(row['MgO %']) else 0.0,
                        "s": float(row['SO3 %']) * 10 * CONV["SO3_S"] if pd.notna(row['SO3 %']) else 0.0
                    }
                }
                ferts_table.insert(new_fert)
        
        st.success("✅ Dünger gespeichert!")
        st.rerun()

# --- TAB 3: PFLANZEN ---
with tab3:
    st.header("Pflanzen & Wachstumsphasen")
    
    # Build table data from database
    table_data = []
    for p in plants_table.all():
        phases = p.get('phases', {})
        for phase_name, nutrient_vals in phases.items():
            table_data.append({
                "Pflanze": p['name'],
                "Phase": phase_name,
                "N (mg/L)": nutrient_vals.get('n', 0.0),
                "P (mg/L)": nutrient_vals.get('p', 0.0),
                "K (mg/L)": nutrient_vals.get('k', 0.0),
                "Ca (mg/L)": nutrient_vals.get('ca', 0.0),
                "Mg (mg/L)": nutrient_vals.get('mg', 0.0),
                "S (mg/L)": nutrient_vals.get('s', 0.0)
            })
    
    # Create editable dataframe
    df = pd.DataFrame(table_data)
    
    st.subheader("Pflanzen & Phasen bearbeiten")
    st.info("Bearbeite die Tabelle direkt. Neue Zeilen werden automatisch hinzugefügt.")
    
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        key="plants_editor"
    )
    
    # Save changes back to database
    if st.button("💾 Änderungen speichern", type="primary", key="save_plants_btn"):
        # Clear old data
        plants_table.truncate()
        
        # Rebuild from edited dataframe
        plants_dict = {}
        for _, row in edited_df.iterrows():
            plant_name = row['Pflanze']
            phase_name = row['Phase']
            
            if plant_name and phase_name:  # Skip empty rows
                if plant_name not in plants_dict:
                    plants_dict[plant_name] = {"name": plant_name, "phases": {}}
                
                plants_dict[plant_name]["phases"][phase_name] = {
                    "n": float(row['N (mg/L)']) if pd.notna(row['N (mg/L)']) else 0.0,
                    "p": float(row['P (mg/L)']) if pd.notna(row['P (mg/L)']) else 0.0,
                    "k": float(row['K (mg/L)']) if pd.notna(row['K (mg/L)']) else 0.0,
                    "ca": float(row['Ca (mg/L)']) if pd.notna(row['Ca (mg/L)']) else 0.0,
                    "mg": float(row['Mg (mg/L)']) if pd.notna(row['Mg (mg/L)']) else 0.0,
                    "s": float(row['S (mg/L)']) if pd.notna(row['S (mg/L)']) else 0.0
                }
        
        # Insert back to database
        for plant in plants_dict.values():
            plants_table.insert(plant)
        
        st.success("✅ Pflanzendaten gespeichert!")
        st.rerun()

# --- TAB 4: LOGS ---
with tab4:
    st.header("Mischungs-Logs")
    
    all_logs = logs_table.all()
    
    if not all_logs:
        st.info("Keine Logs vorhanden. Erstelle eine Mischung und logge sie!")
    else:
        # Display logs in reverse order (newest first)
        for log in reversed(all_logs):
            with st.expander(f"📅 {log['timestamp']} - {log['plant']} ({log['phase']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Pflanze:** {log['plant']}")
                    st.write(f"**Phase:** {log['phase']}")
                    st.write(f"**Wasser:** {log['water_type']}")
                    st.write(f"**Menge:** {log['liters']} L")
                
                with col2:
                    st.write(f"**Datum/Zeit:** {log['timestamp']}")
                    st.write(f"**Kosten:** {log['total_cost']} €")
                
                st.write("**Düngemittel:**")
                fert_log_df = pd.DataFrame(log['fertilizers'])
                st.table(fert_log_df)