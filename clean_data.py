import numpy as np
import pandas as pd


def detect_non_outliers(values, alpha=0.01):
    log_values = np.log1p(values)
    lower_bound = log_values.quantile(alpha)
    upper_bound = log_values.quantile(1-alpha)
    return (log_values >= lower_bound) & (log_values <= upper_bound)


target_clients = pd.read_csv("data/prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv")
unique_targets = target_clients[["nit_enmascarado", "num_oblig_enmascarado"]].drop_duplicates()

demographic_columns = ["nit_enmascarado", "genero_cli", "edad_cli", "estado_civil", "tipo_vivienda", "num_hijos",
                       "personas_dependientes", "total_ing", "tot_activos", "tot_pasivos",
                       "segm", "region_of", "egresos_mes", "tot_patrimonio"]
demographic = pd.read_csv("data/prueba_op_master_customer_data_enmascarado_completa.csv")

demographic["year"] = demographic["year"].apply(lambda year: str(year).zfill(4))
demographic["month"] = demographic["month"].apply(lambda month: str(month).zfill(2))
demographic["ingestion_day"] = demographic["ingestion_day"].apply(lambda day: str(day).zfill(2))
demographic["ingestion"] = demographic["year"] + demographic["month"] + demographic["ingestion_day"]

# Find the rows with the latest 'ingestion' for each 'nit_enmascarado'
demographic = demographic.loc[
    demographic.groupby("nit_enmascarado")["ingestion"].idxmax()
]

demographic = demographic[demographic_columns]

# Clean gender
most_frequent_value = demographic['genero_cli'].mode()[0]
demographic['genero_cli'] = demographic['genero_cli'].fillna(most_frequent_value)

# Clean age
mean_age = demographic["edad_cli"].mean()
demographic["edad_cli"] = demographic["edad_cli"].apply(lambda age: age if 18 <= age <= 90 else mean_age)


# Clean civil status
civil_status_mapping = {
    'SOLTERO': 'SOLTERO',
    'DIVORCIADO': 'SOLTERO',
    'VIUDO': 'SOLTERO',
    'OTRO': 'SOLTERO',
    'UNION LIBRE': 'PAREJA',
    'CASADO': 'PAREJA',
    'NO INFORMA': 'NO INFORMA'
}
demographic['estado_civil'] = demographic['estado_civil'].replace(civil_status_mapping)

# Clean home type
demographic["tipo_vivienda"] = demographic["tipo_vivienda"].fillna("DESCONOCIDO")
demographic["tipo_vivienda"] = demographic["tipo_vivienda"].replace({"NO INFORMA": "DESCONOCIDO"})

# Clean number of childs
demographic["tiene_hijos"] = demographic["num_hijos"].apply(lambda n_childs: 1 if n_childs > 0 else 0)
demographic.drop("num_hijos", inplace=True, axis=1)

# Clean dependants
demographic["tiene_dependientes"] = demographic["personas_dependientes"].apply(lambda n_dep: 1 if n_dep > 0 else 0)
demographic.drop("personas_dependientes", inplace=True, axis=1)

# Clean income
demographic = demographic[demographic["total_ing"] > 0]
mask = detect_non_outliers(demographic["total_ing"], alpha=0.02)
demographic = demographic[mask]

# Clean assets
demographic = demographic[demographic["tot_activos"] > 0]
mask = detect_non_outliers(demographic["tot_activos"], alpha=0.01)
demographic = demographic[mask]

# Clean liabilities
demographic["tiene_pasivos"] = demographic["tot_pasivos"].apply(lambda value: 1 if value > 0 else 0)
demographic.drop(columns="tot_pasivos", inplace=True)

# Clean Region
demographic["region_of"] = demographic["region_of"].replace({
    "DIRECCIÓN GENERAL": "ANTIOQUIA",
    "BANCO": "ANTIOQUIA",
    "BANCO DE COLOMBIA": "ANTIOQUIA"
})

# Clean expenses
num_buckets = 5
demographic['egresos_mes'] = pd.qcut(demographic['egresos_mes'],
                                     q=num_buckets,
                                     labels=[f"Q{i+1}" for i in range(num_buckets)])

# Clean capital
demographic = demographic[demographic["tot_patrimonio"] > 0]


def add_missing_dates(data_df, id_columns, period_column, target_months):

    date_range = pd.DataFrame({period_column: target_months})
    unique_pairs = data_df[id_columns].drop_duplicates()
    all_combinations = unique_pairs.merge(date_range, how='cross')
    data_df = all_combinations.merge(data_df, on=id_columns + [period_column],
                                     how='left')
    data_df.sort_values(by=id_columns + [period_column], inplace=True)
    return data_df


def extend_in_time(data_df, id_columns, columns_to_transpose, window_size=3):
    result = []
    for group_id, group_data in data_df.groupby(by=id_columns):
        group_data.reset_index(inplace=True, drop=True)

        for i in range(len(group_data) - window_size + 1):
            data_slice = group_data.iloc[i: i + window_size]
            window_data = {name: value for name, value in zip(id_columns, group_id)}
            window_data.update({"window": i})
            for column in columns_to_transpose:
                column_values = data_slice[column].values
                column_data = {
                    f"{column}_{len(column_values) - value_ind - 1}": column_values[value_ind]
                    for value_ind in range(len(column_values))
                }
                window_data.update(column_data)
            result.append(window_data)

    result = pd.DataFrame(result)
    return result

# Clean model outputs

print("Cleaning model outputs")

months = [202305, 202306, 202307, 202308, 202309, 202310, 202311, 202312]
bc_models_columns = ["nit_enmascarado", "num_oblig_enmascarado", "fecha_corte", "prob_propension",
                     "prob_alrt_temprana", "prob_auto_cura"]

bc_models = pd.read_csv("data/prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.csv")
bc_models = bc_models.merge(unique_targets, on=["nit_enmascarado", "num_oblig_enmascarado"])
bc_models = bc_models[bc_models_columns]

bc_models["prob_auto_cura"].dropna(inplace=True)
bc_models["prob_alrt_temprana"].dropna(inplace=True)


bc_models = bc_models[bc_models["fecha_corte"].isin(months)]
bc_models_parsed = add_missing_dates(data_df=bc_models, id_columns=["nit_enmascarado", "num_oblig_enmascarado"],
                                     period_column="fecha_corte", target_months=months)
bc_models_parsed = extend_in_time(data_df=bc_models_parsed, id_columns=["nit_enmascarado", "num_oblig_enmascarado"],
                                  columns_to_transpose=['prob_propension', 'prob_alrt_temprana', 'prob_auto_cura'])


# Clean payment hist
print("Clean payment hist")

payment_columns = ["nit_enmascarado", "num_oblig_enmascarado", "fecha_corte", "pago_total"]
payment_hist = pd.read_csv("data/prueba_op_maestra_cuotas_pagos_mes_hist_enmascarado_completa.csv")[payment_columns]
payment_hist["fecha_corte"] = payment_hist["fecha_corte"].apply(lambda date: int(date/100))
payment_hist = payment_hist[payment_hist["fecha_corte"].isin(months)]

bad_nits = [372957, 456315, 468511, 606094]
payment_hist = payment_hist[~payment_hist["nit_enmascarado"].isin(bad_nits)]

payment_hist = payment_hist.merge(unique_targets, on=["nit_enmascarado", "num_oblig_enmascarado"])

payment_hist_parsed = add_missing_dates(
    data_df=payment_hist, id_columns=["nit_enmascarado", "num_oblig_enmascarado"],
    period_column="fecha_corte", target_months=months
)
payment_hist_parsed = extend_in_time(data_df=payment_hist_parsed,
                                     id_columns=["nit_enmascarado", "num_oblig_enmascarado"],
                                     columns_to_transpose=["pago_total"])

# Clean product data

product_data_columns = ["nit_enmascarado", "num_oblig_enmascarado", "producto", "aplicativo"]
product_data = pd.read_csv("data/prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv")[product_data_columns]
product_data.drop_duplicates(inplace=True, subset=["nit_enmascarado", "num_oblig_enmascarado"])

product_type_mapping = {
    "TARJETA DE CREDITO": "consumo sin garantia",
    "LIBRE INVERSION": "consumo sin garantia",
    "ROTATIVOS": "consumo sin garantia",
    "CREDIAGIL": "consumo sin garantia",
    "CREDIPAGO": "consumo sin garantia",
    "CREDITO A LA MANO": "consumo sin garantia",
    "CREDITOS DE CONSUMO": "consumo sin garantia",
    "CARTERA ORDINARIA": "consumo sin garantia",
    "HIPOTECARIO VIVIENDA": "consumo con garantia",
    "LEASING HABITACIONAL": "consumo con garantia",
    "OTROS HIPOTECARIO": "consumo con garantia",
    "CREDITO HIPOTECARIO": "consumo con garantia",
    "CARTERA MICROCREDITO": "empresarial",
    "MICROCREDITO": "empresarial",
    "TESORERIA": "empresarial",
    "SOBREGIRO": "empresarial",
    "LEASING": "empresarial",
    "LIBRANZA": "nomina",
    "LIBRANZA EX EMPLEADOS": "nomina",
    "REESTRUCTURADO": "especial",
    "Titularizada": "especial",
    "VENTA DIGITAL": "digital",
    "CUENTA CORRIENTE": "especial",
    "TARJETAS DE CREDITO": "consumo sin garantia"
}

product_data["producto"] = product_data["producto"].replace(product_type_mapping)
product_data = product_data[~product_data["producto"].isin(["especial", "digital"])]

product_data["aplicativo"] = product_data["aplicativo"].replace({"D": "4", "3": "4"})

# Get labels
labels = pd.read_csv("data/prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv")[[
    "nit_enmascarado",
    "num_oblig_orig_enmascarado",
    "num_oblig_enmascarado",
    "fecha_var_rpta_alt",
    "var_rpta_alt"
]]

labels_dates = labels["fecha_var_rpta_alt"].unique()
labels_dates.sort()
window_map = {int(label_date): i for i, label_date in enumerate(labels_dates)}
labels["window"] = labels["fecha_var_rpta_alt"].replace(window_map)


# Merge dataframes


features = bc_models_parsed.merge(payment_hist_parsed,
                                  on=["nit_enmascarado", "num_oblig_enmascarado", "window"],
                                  how="inner")
print("features shape:", features[features.window == 0].shape[0])
features = features.merge(demographic, on=["nit_enmascarado"], how="left")
print("features shape:", features[features.window == 0].shape[0])
features = features.merge(product_data, on=["nit_enmascarado", "num_oblig_enmascarado"], how="left")
print("features shape:", features[features.window == 0].shape[0])

test_features = features[features["window"] == 5]

features = features.merge(labels,
                          on=["nit_enmascarado", "num_oblig_enmascarado", "window"],
                          how="inner")
print("features shape:", features[features.window == 0].shape[0])



import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
target_feature = "prob_alrt_temprana_0"
plt.hist(
    np.log1p(features[features['var_rpta_alt'] == 1][target_feature]),
    bins=30, alpha=0.7, label='Label = 1', color='orchid'
)
plt.hist(
    np.log1p(features[features['var_rpta_alt'] == 0][target_feature]),
    bins=30, alpha=0.7, label='Label = 0', color='turquoise'
)

plt.xlabel(target_feature)
plt.ylabel('Frequency')
plt.title(f'Histogram of {target_feature} by var_rpta_alt')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show(block=True)