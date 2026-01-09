import pandas as pd
import numpy as np
import altair as alt


class NutrientProfile:
    """Hilfsfunktionen zum Erzeugen von Vergleichs- und Kreisdiagrammen für Nährstoffprofile."""

    NUTRIENT_KEYS = ['n', 'p', 'k', 'ca', 'mg', 's']
    NUTRIENT_LABELS = ['N', 'P', 'K', 'Ca', 'Mg', 'S']

    @staticmethod
    def layered_mixture_vs_target(netto_req, matrix, amounts, fert_names):
        """Erzeugt ein Altair-Layer-Chart: gestapelte Beiträge pro Dünger + grüne Zielbalken.

        netto_req: array-like (6,) Ziel netto mg/L
        matrix: numpy array shape (6, n_ferts) mit mg/ml Werten
        amounts: array-like (n_ferts,) ml/L
        fert_names: list of fert names
        """
        nutrient_labels = NutrientProfile.NUTRIENT_LABELS

        amounts = np.array(amounts)
        contrib = matrix * amounts  # broadcasting -> (nutrients, ferts)

        # Melt contributions
        rows = []
        for i, nutr in enumerate(nutrient_labels):
            for j, fname in enumerate(fert_names):
                val = float(contrib[i, j])
                if val > 1e-9:
                    rows.append({"Nährstoff": nutr, "Dünger": fname, "value": val, "group": "Mischung"})
        mix_df = pd.DataFrame(rows)

        targ_rows = [{"Nährstoff": nutrient_labels[i], "Dünger": "Ziel", "value": float(netto_req[i]), "group": "Ziel"} for i in range(len(nutrient_labels))]
        targ_df = pd.DataFrame(targ_rows)

        mix_chart = alt.Chart(mix_df).mark_bar().encode(
            x=alt.X('Nährstoff:N', title='Nährstoff'),
            y=alt.Y('value:Q', title='mg/L', stack='zero'),
            color=alt.Color('Dünger:N', legend=alt.Legend(title='Dünger')),
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
    def pie_df_from_matrix(matrix, amounts, fert_names):
        """Gibt ein DataFrame mit Gesamtsummen pro Dünger zurück und Prozent-Anteil."""
        amounts = np.array(amounts)
        contrib = matrix * amounts  # (nutrients, ferts)
        fert_totals = contrib.sum(axis=0) if contrib.size else np.array([])
        rows = []
        for j, fname in enumerate(fert_names):
            val = float(fert_totals[j]) if j < len(fert_totals) else 0.0
            if val > 1e-9:
                rows.append({"Dünger": fname, "value": val})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df['pct'] = df['value'] / df['value'].sum() * 100
        return df

    @staticmethod
    def pie_chart_from_df(pie_df):
        if pie_df.empty:
            return None
        return alt.Chart(pie_df).mark_arc(innerRadius=40).encode(
            theta=alt.Theta('value:Q', title='mg/L'),
            color=alt.Color('Dünger:N', legend=alt.Legend(title='Dünger')),
            tooltip=[alt.Tooltip('Dünger:N'), alt.Tooltip('value:Q', format='.2f'), alt.Tooltip('pct:Q', format='.1f')]
        ).properties(height=300)
