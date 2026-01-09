import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import altair as alt
from tinydb import TinyDB, Query
from scipy.optimize import minimize

# --- KONFIGURATION & DB ---
st.set_page_config(page_title="Hydro Optimizer", layout="wide")
db = TinyDB('db.json')
plants_table = db.table('plants')
ferts_table = db.table('fertilizers')

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
def run_optimization(target_profile, available_ferts):
    """Berechnet die optimale Menge ml/L pro Dünger."""
    nutrient_keys = ['n', 'p', 'k', 'ca', 'mg', 's']
    
    # Netto-Bedarf (Ziel - Wasser)
    target_netto = np.array([max(0, target_profile[k] - WATER_QUALITY[k]) for k in nutrient_keys])
    
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
tab1, tab2, tab3 = st.tabs(["🧮 Optimierung", "🧪 Düngemittel", "🌱 Pflanzen"])

# --- TAB 1: OPTIMIERUNG ---
with tab1:
    st.header("Mischung berechnen")
    
    all_plants = plants_table.all()
    all_ferts = ferts_table.all()

    if not all_plants or not all_ferts:
        st.warning("Bitte lege zuerst Pflanzen und Düngemittel in den anderen Tabs an!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            p_name = st.selectbox("Pflanze wählen", [p['name'] for p in all_plants])
            plant = next(p for p in all_plants if p['name'] == p_name)
            phase = st.selectbox("Wachstumsphase", list(plant['phases'].keys()))
            target_vals = plant['phases'][phase]

        with col2:
            st.info(f"**Basis: Leitungswasser Dagersheim**\n(Ca: {WATER_QUALITY['ca']}, Mg: {WATER_QUALITY['mg']} mg/L)")
            with st.expander("Zielprofil Details (mg/L)"):
                st.write(target_vals)

        if st.button("🚀 Besten Mix berechnen", type="primary"):
            amounts, netto_req, achieved = run_optimization(target_vals, all_ferts)
            
            st.subheader("Ergebnis: Dosierung für 10 Liter Wasser")
            res_data = []
            total_cost = 0
            for i, fert in enumerate(all_ferts):
                if amounts[i] > 0.01:
                    cost = amounts[i] * 10 * fert.get('price_per_ml', 0)
                    res_data.append({
                        "Dünger": fert['name'],
                        "ml pro 10L": round(amounts[i] * 10, 2),
                        "Kosten (€/10L)": f"{cost:.2f} €"
                    })
                    total_cost += cost
            
            st.table(res_data)
            st.metric("Gesamtkosten pro 10L", f"{total_cost:.2f} €")

            # Vergleichschart (grün = Ziel, grau = Mischung)
            st.subheader("Profil-Abgleich (Ziel vs. Erreicht)")
            comparison_df = pd.DataFrame({
                "Nährstoff": ['N', 'P', 'K', 'Ca', 'Mg', 'S'],
                "Bedarf (Netto)": netto_req,
                "Erreicht (Dünger)": achieved
            })

            # In langes Format für gruppierte Balken umwandeln
            melt = comparison_df.melt(id_vars='Nährstoff', var_name='type', value_name='value')
            melt['type'] = melt['type'].map({'Bedarf (Netto)': 'Ziel', 'Erreicht (Dünger)': 'Mischung'})

            chart = alt.Chart(melt).mark_bar().encode(
                x=alt.X('Nährstoff:N', title='Nährstoff'),
                y=alt.Y('value:Q', title='mg/L'),
                color=alt.Color('type:N', scale=alt.Scale(domain=['Ziel', 'Mischung'], range=['#2ca02c', '#b0b0b0']), legend=alt.Legend(title='')),
                xOffset='type:N',
                tooltip=['Nährstoff', 'type', alt.Tooltip('value:Q', format='.2f')]
            ).properties(height=320)

            st.altair_chart(chart, use_container_width=True)

# --- TAB 2: DÜNGEMITTEL ---
with tab2:
    st.header("Düngemittel-Datenbank")
    
    with st.expander("➕ Neuen Dünger hinzufügen / Oxid-Rechner"):
        with st.form("fert_form"):
            name = st.text_input("Name des Düngers")
            price = st.number_input("Preis pro ml (€)", format="%.4f", value=0.015)
            
            st.write("Inhaltsstoffe (in % oder mg/ml eingeben)")
            c1, c2, c3 = st.columns(3)
            n_val = c1.number_input("N (Gesamtstickstoff)")
            p2o5 = c2.number_input("P2O5 (Phosphorpentoxid)")
            k2o = c3.number_input("K2O (Kaliumoxid)")
            cao = c1.number_input("CaO (Calciumoxid)")
            mgo = c2.number_input("MgO (Magnesiumoxid)")
            so3 = c3.number_input("SO3 (Schwefeltrioxid)")
            
            if st.form_submit_button("Speichern"):
                # Umrechnung in elementare mg/ml (Annahme: 1% = 10mg/ml bei flüssig)
                new_fert = {
                    "name": name,
                    "price_per_ml": price,
                    "composition": {
                        "n": n_val * 10,
                        "p": p2o5 * 10 * CONV["P2O5_P"],
                        "k": k2o * 10 * CONV["K2O_K"],
                        "ca": cao * 10 * CONV["CaO_Ca"],
                        "mg": mgo * 10 * CONV["MgO_Mg"],
                        "s": so3 * 10 * CONV["SO3_S"]
                    }
                }
                ferts_table.insert(new_fert)
                st.success(f"{name} hinzugefügt!")
                st.rerun()

        # Liste anzeigen (mit Editiermöglichkeit)
    for f in ferts_table.all():
        col_f1, col_f2, col_f3 = st.columns([4, 1, 1])
        col_f1.write(f"**{f['name']}** ({f.get('price_per_ml')} €/ml)")
        if col_f2.button("Löschen", key=f"del_{f['name']}"):
            ferts_table.remove(Query().name == f['name'])
            st.rerun()
        if col_f3.button("Bearbeiten", key=f"edit_{f['name']}"):
            st.session_state[f"edit_fert_{f['name']}"] = True

        if st.session_state.get(f"edit_fert_{f['name']}", False):
            with st.expander(f"Bearbeite {f['name']}", expanded=True):
                with st.form(f"fert_edit_form_{f['name']}"):
                    ename = st.text_input("Name des Düngers", value=f['name'], key=f"ename_{f['name']}")
                    eprice = st.number_input("Preis pro ml (€)", format="%.4f", value=f.get('price_per_ml', 0.0), key=f"eprice_{f['name']}")
                    st.write("Inhaltsstoffe (elementar, mg/ml)")
                    c1, c2, c3 = st.columns(3)
                    en = c1.number_input("N (mg/ml)", value=f.get('composition', {}).get('n', 0.0), key=f"en_{f['name']}")
                    ep = c2.number_input("P (mg/ml)", value=f.get('composition', {}).get('p', 0.0), key=f"ep_{f['name']}")
                    ek = c3.number_input("K (mg/ml)", value=f.get('composition', {}).get('k', 0.0), key=f"ek_{f['name']}")
                    ca = c1.number_input("Ca (mg/ml)", value=f.get('composition', {}).get('ca', 0.0), key=f"eca_{f['name']}")
                    mg = c2.number_input("Mg (mg/ml)", value=f.get('composition', {}).get('mg', 0.0), key=f"emg_{f['name']}")
                    s = c3.number_input("S (mg/ml)", value=f.get('composition', {}).get('s', 0.0), key=f"es_{f['name']}")
                    if st.form_submit_button("Speichern", key=f"save_fert_{f['name']}"):
                        new_comp = {"n": en, "p": ep, "k": ek, "ca": ca, "mg": mg, "s": s}
                        ferts_table.update({'name': ename, 'price_per_ml': eprice, 'composition': new_comp}, Query().name == f['name'])
                        st.success("Dünger aktualisiert")
                        st.session_state.pop(f"edit_fert_{f['name']}", None)
                        st.rerun()

# --- TAB 3: PFLANZEN ---
with tab3:
    st.header("Pflanzen & Wachstumsphasen")
    
    with st.form("plant_form"):
        p_name = st.text_input("Name der Pflanze (z.B. Tomate)")
        st.write("Zielwerte für die Phase (mg/L)")
        phase_name = st.text_input("Name der Phase (z.B. Bloom)")
        
        c1, c2, c3 = st.columns(3)
        tn = c1.number_input("Ziel N")
        tp = c2.number_input("Ziel P")
        tk = c3.number_input("Ziel K")
        tca = c1.number_input("Ziel Ca")
        tmg = c2.number_input("Ziel Mg")
        ts = c3.number_input("Ziel S")
        
        if st.form_submit_button("Pflanze/Phase speichern"):
            existing = plants_table.search(Query().name == p_name)
            if existing:
                phases = existing[0]['phases']
                phases[phase_name] = {"n": tn, "p": tp, "k": tk, "ca": tca, "mg": tmg, "s": ts}
                plants_table.update({'phases': phases}, Query().name == p_name)
            else:
                plants_table.insert({
                    "name": p_name,
                    "phases": {phase_name: {"n": tn, "p": tp, "k": tk, "ca": tca, "mg": tmg, "s": ts}}
                })
            st.success("Gespeichert!")
            st.rerun()

    # Liste der Pflanzen (mit Editiermöglichkeit)
    for p in plants_table.all():
        st.write(f"### {p['name']}")
        st.json(p['phases'])
        colp1, colp2 = st.columns([4, 1])
        if colp2.button(f"{p['name']} löschen", key=f"delplant_{p['name']}"):
            plants_table.remove(Query().name == p['name'])
            st.rerun()

        if st.button(f"Bearbeiten {p['name']}", key=f"edit_plant_btn_{p['name']}"):
            st.session_state[f"edit_plant_{p['name']}"] = True

        if st.session_state.get(f"edit_plant_{p['name']}", False):
            with st.expander(f"Bearbeite {p['name']}", expanded=True):
                with st.form(f"plant_edit_form_{p['name']}"):
                    new_name = st.text_input("Pflanzenname", value=p['name'], key=f"pname_{p['name']}")
                    phases = p.get('phases', {})
                    phase_options = list(phases.keys())
                    sel = st.selectbox("Phase wählen zum Bearbeiten", options=(phase_options + ["--Neu--"]) if phase_options else ["--Neu--"], key=f"phase_sel_{p['name']}")
                    phase_name = st.text_input("Phasenname", value=(sel if sel != "--Neu--" else ""), key=f"phase_name_{p['name']}")
                    c1, c2, c3 = st.columns(3)
                    cur = phases.get(sel, {}) if sel != "--Neu--" else {}
                    tn = c1.number_input("Ziel N", value=cur.get('n', 0.0), key=f"tn_{p['name']}")
                    tp = c2.number_input("Ziel P", value=cur.get('p', 0.0), key=f"tp_{p['name']}")
                    tk = c3.number_input("Ziel K", value=cur.get('k', 0.0), key=f"tk_{p['name']}")
                    tca = c1.number_input("Ziel Ca", value=cur.get('ca', 0.0), key=f"tca_{p['name']}")
                    tmg = c2.number_input("Ziel Mg", value=cur.get('mg', 0.0), key=f"tmg_{p['name']}")
                    ts = c3.number_input("Ziel S", value=cur.get('s', 0.0), key=f"ts_{p['name']}")
                    if st.form_submit_button("Speichern", key=f"save_plant_{p['name']}"):
                        updated_phases = dict(phases)
                        if phase_name:
                            updated_phases[phase_name] = {"n": tn, "p": tp, "k": tk, "ca": tca, "mg": tmg, "s": ts}
                        plants_table.update({'name': new_name, 'phases': updated_phases}, Query().name == p['name'])
                        st.success("Pflanze/Phase gespeichert")
                        st.session_state.pop(f"edit_plant_{p['name']}", None)
                        st.rerun()