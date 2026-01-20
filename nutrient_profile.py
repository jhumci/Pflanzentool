import pandas as pd
import numpy as np
import altair as alt


class NutrientProfile:
    """Hilfsfunktionen zum Erzeugen von Vergleichs- und Kreisdiagrammen für Nährstoffprofile."""

    NUTRIENT_KEYS = ['n', 'p', 'k', 'ca', 'mg', 's']
    NUTRIENT_LABELS = ['N', 'P', 'K', 'Ca', 'Mg', 'S']

    @staticmethod
    def layered_mixture_vs_target(netto_req, matrix, amounts, fert_names, water_profile=None):
        """Erzeugt ein Altair-Layer-Chart: Wasser + gestapelte Beiträge pro Dünger + grüne Zielbalken.

        netto_req: array-like (6,) Ziel netto mg/L (ohne Wasser)
        matrix: numpy array shape (6, n_ferts) mit mg/ml Werten
        amounts: array-like (n_ferts,) ml/L
        fert_names: list of fert names
        water_profile: dict mit 'n', 'p', 'k', 'ca', 'mg', 's' Werten (optional)
        """
        nutrient_labels = NutrientProfile.NUTRIENT_LABELS

        amounts = np.array(amounts)
        contrib = matrix * amounts  # broadcasting -> (nutrients, ferts)

        # Melt contributions
        rows = []
        
        # Wasser hinzufügen, falls vorhanden
        if water_profile:
            for i, nutr in enumerate(nutrient_labels):
                water_val = water_profile.get(['n', 'p', 'k', 'ca', 'mg', 's'][i], 0)
                if water_val > 1e-9:
                    rows.append({"Nährstoff": nutr, "Dünger": "Wasser", "value": float(water_val), "group": "Mischung"})
        
        # Dünger-Beiträge
        for i, nutr in enumerate(nutrient_labels):
            for j, fname in enumerate(fert_names):
                val = float(contrib[i, j])
                if val > 1e-9:
                    rows.append({"Nährstoff": nutr, "Dünger": fname, "value": val, "group": "Mischung"})
        mix_df = pd.DataFrame(rows)

        # Zielwerte (total = Wasser + Dünger)
        targ_rows = []
        for i in range(len(nutrient_labels)):
            total_target = float(netto_req[i])
            if water_profile:
                total_target += water_profile.get(['n', 'p', 'k', 'ca', 'mg', 's'][i], 0)
            targ_rows.append({"Nährstoff": nutrient_labels[i], "Dünger": "Ziel", "value": total_target, "group": "Ziel"})
        targ_df = pd.DataFrame(targ_rows)

        mix_chart = alt.Chart(mix_df).mark_bar().encode(
            x=alt.X('Nährstoff:N', title='Nährstoff'),
            y=alt.Y('value:Q', title='mg/L', stack='zero'),
            color=alt.Color('Dünger:N', legend=alt.Legend(title='Komponente')),
            xOffset=alt.XOffset('group:N'),
            tooltip=['Nährstoff', 'Dünger', alt.Tooltip('value:Q', format='.2f')]
        )

        target_chart = alt.Chart(targ_df).mark_bar(color='#2ca02c').encode(
            x=alt.X('Nährstoff:N'),
            y=alt.Y('value:Q'),
            xOffset=alt.XOffset('group:N'),
            tooltip=['Nährstoff', alt.Tooltip('value:Q', format='.2f')]
        )

        return alt.layer(mix_chart, target_chart).properties(height=360)

    @staticmethod
    def pie_df_from_matrix(matrix, amounts, fert_names, water_profile=None):
        """Gibt ein DataFrame mit Gesamtsummen pro Dünger + Wasser zurück und Prozent-Anteil."""
        amounts = np.array(amounts)
        contrib = matrix * amounts  # (nutrients, ferts)
        fert_totals = contrib.sum(axis=0) if contrib.size else np.array([])
        rows = []
        
        # Dünger
        for j, fname in enumerate(fert_names):
            val = float(fert_totals[j]) if j < len(fert_totals) else 0.0
            if val > 1e-9:
                rows.append({"Komponente": fname, "value": val})
        
        # Wasser hinzufügen
        if water_profile:
            water_total = sum(water_profile.get(k, 0) for k in ['n', 'p', 'k', 'ca', 'mg', 's'])
            if water_total > 1e-9:
                rows.append({"Komponente": "Wasser", "value": float(water_total)})
        
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df['pct'] = df['value'] / df['value'].sum() * 100
        return df

    @staticmethod
    def pie_df_volume(amounts, fert_names, liters):
        """Gibt ein DataFrame mit Volumenanteilen zurück (ml Dünger vs ml Wasser)."""
        amounts = np.array(amounts)
        total_ml = liters * 1000  # in ml
        fert_ml = amounts * liters  # ml pro Liter * Liter
        total_fert_ml = np.sum(fert_ml)
        water_ml = total_ml - total_fert_ml
        
        rows = []
        for j, fname in enumerate(fert_names):
            val = float(fert_ml[j])
            if val > 1e-9:
                rows.append({"Komponente": fname, "volume": val})
        
        if water_ml > 1e-9:
            rows.append({"Komponente": "Wasser", "volume": float(water_ml)})
        
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df['pct'] = df['volume'] / df['volume'].sum() * 100
        return df

    @staticmethod
    def pie_df_volume_no_water(amounts, fert_names):
        """Gibt ein DataFrame mit Volumenanteilen NUR der Dünger zurück (ohne Wasser)."""
        amounts = np.array(amounts)
        
        rows = []
        for j, fname in enumerate(fert_names):
            val = float(amounts[j])
            if val > 1e-9:
                rows.append({"Komponente": fname, "volume": val})
        
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df['pct'] = df['volume'] / df['volume'].sum() * 100
        return df

    @staticmethod
    def pie_chart_volume(pie_df, title="Volumenanteile"):
        """Erzeugt ein Altair Pie Chart für Volumenanteile."""
        if pie_df.empty:
            return None
        
        return alt.Chart(pie_df).mark_arc(innerRadius=40).encode(
            theta=alt.Theta('volume:Q'),
            color=alt.Color('Komponente:N', legend=alt.Legend(title='Komponente')),
            tooltip=[alt.Tooltip('Komponente:N'), alt.Tooltip('volume:Q', format='.2f'), alt.Tooltip('pct:Q', format='.1f')]
        ).properties(height=300, title=title)

    @staticmethod
    def pie_chart_from_df(pie_df):
        if pie_df.empty:
            return None
        return alt.Chart(pie_df).mark_arc(innerRadius=40).encode(
            theta=alt.Theta('value:Q', title='mg/L'),
            color=alt.Color('Komponente:N', legend=alt.Legend(title='Komponente')),
            tooltip=[alt.Tooltip('Komponente:N'), alt.Tooltip('value:Q', format='.2f'), alt.Tooltip('pct:Q', format='.1f')]
        ).properties(height=300)
