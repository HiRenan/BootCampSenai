import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

# --- Configuração da Página ---
st.set_page_config(
    layout="wide", 
    page_title="Análise Preditiva de Defeitos em Chapas de Aço", 
    page_icon="🏭" # Ícone 
)

# Estilo CSS 
st.markdown("""
<style>
    /* Cores e Fontes Globais */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        background-color: #eef2f7; /* Um cinza azulado muito claro */
        color: #2c3e50; /* Cinza escuro para texto principal */
    }
    /* Títulos Principais */
    h1 {
        color: #1a5276; /* Azul petróleo escuro */
        text-align: center;
        padding-bottom: 25px;
        font-weight: bold;
    }
    /* Cabeçalhos de Secção */
    h2 {
        color: #2980b9; /* Azul mais vibrante */
        border-bottom: 3px solid #2980b9;
        padding-bottom: 12px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    /* Subcabeçalhos */
    h3 {
        color: #3498db; /* Azul claro */
        margin-top: 25px;
    }
    /* DataFrames */
    .stDataFrame table {
        font-size: 14px;
        border: 1px solid #bdc3c7; /* Borda suave */
    }
    .stDataFrame th {
        background-color: #34495e; /* Cinza ardósia */
        color: white;
        font-size: 16px;
        font-weight: bold;
    }
    .stDataFrame td {
        border-bottom: 1px solid #ecf0f1; /* Linhas de separação suaves */
    }
    /* Botões */
    .stButton>button {
        background-color: #27ae60; /* Verde esmeralda */
        color: white;
        border-radius: 8px;
        padding: 12px 25px;
        border: none;
        font-weight: bold;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #229954; /* Verde mais escuro no hover */
    }
    /* Barra Lateral */
    .css-1d391kg {
        background-color: #d6eaf8; /* Azul muito claro para a barra lateral */
        border-right: 1px solid #aed6f1;
    }
    .css-1d391kg .stRadio > label > div:first-child{
        font-weight: bold;
        color: #1a5276;
    }
    /* Containers e Expanders */
    .stExpander {
        border: 1px solid #aed6f1;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .stExpander header {
        background-color: #f2f8fd;
        font-weight: bold;
        color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# --- Funções de Pré-processamento ---
def preprocess_input_for_prediction(input_data, scaler, train_feature_columns, train_medians):
    processed_input = pd.DataFrame(columns=train_feature_columns)
    for col in input_data.columns:
        if col in processed_input.columns:
            processed_input[col] = input_data[col]
    for col in train_feature_columns:
        if col not in input_data.columns or pd.isnull(input_data[col].iloc[0]):
            processed_input[col] = train_medians.get(col, 0) # Usar .get() para segurança
    if 'peso_da_placa' in processed_input.columns: # Embora já deva ter sido removido
        processed_input = processed_input.drop(columns=['peso_da_placa'])
    final_input_cols = [col for col in train_feature_columns if col != 'peso_da_placa']
    processed_input = processed_input[final_input_cols]
    if scaler:
        try:
            processed_input_scaled = scaler.transform(processed_input)
            return pd.DataFrame(processed_input_scaled, columns=final_input_cols)
        except Exception as e:
            st.error(f"Erro ao aplicar o scaler: {e}")
            return None
    st.error("Scaler não foi carregado corretamente.")
    return None

# --- Carregar Dados e Modelos ---
@st.cache_data
def load_main_data():
    try:
        df = pd.read_csv("data/bootcamp_train.csv")
        # Limpeza básica de nomes de colunas
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Ficheiro 'bootcamp_train.csv' não encontrado. Verifique o caminho.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_resource
def load_model_components():
    try:
        train_df_for_model = pd.read_csv("data/bootcamp_train.csv")
        train_df_for_model.columns = train_df_for_model.columns.str.strip()

        falha_cols_model = [f'falha_{i}' for i in range(1, 7)] + ['falha_outros']
        map_sim_nao_model = {'Sim': 1, 'sim': 1, 'True': 1, True: 1, 'Não': 0, 'nao': 0, 'Nao': 0, 'False': 0, False: 0}
        
        for col in falha_cols_model:
            if train_df_for_model[col].dtype == 'object' or pd.api.types.is_string_dtype(train_df_for_model[col]):
                train_df_for_model[col] = train_df_for_model[col].apply(lambda x: map_sim_nao_model.get(str(x).strip(), 0) if pd.notnull(x) else 0)
            train_df_for_model[col] = train_df_for_model[col].astype(int)

        for col_aco in ['tipo_do_aço_A300', 'tipo_do_aço_A400']:
            if col_aco in train_df_for_model.columns:
                if train_df_for_model[col_aco].dtype == 'object':
                    train_df_for_model[col_aco] = train_df_for_model[col_aco].map(map_sim_nao_model).fillna(0).astype(int)
                elif pd.api.types.is_bool_dtype(train_df_for_model[col_aco]):
                    train_df_for_model[col_aco] = train_df_for_model[col_aco].astype(int)
                if train_df_for_model[col_aco].isnull().any():
                    train_df_for_model[col_aco] = train_df_for_model[col_aco].fillna(0).astype(int)
        
        numerical_cols_model = train_df_for_model.select_dtypes(include=np.number).columns.tolist()
        cols_to_impute_model = [col for col in numerical_cols_model if col != 'id' and col not in falha_cols_model]
        
        train_medians_model = {}
        for col in cols_to_impute_model:
            if train_df_for_model[col].isnull().any():
                median_val = train_df_for_model[col].median()
                train_medians_model[col] = median_val
                train_df_for_model[col] = train_df_for_model[col].fillna(median_val)
            elif col not in train_medians_model: 
                 train_medians_model[col] = train_df_for_model[col].median()
        
        if 'peso_da_placa' in train_df_for_model.columns:
            train_df_for_model = train_df_for_model.drop(columns=['peso_da_placa'])
        
        features_to_scale_model = [col for col in train_df_for_model.columns if train_df_for_model[col].dtype in [np.int64, np.float64, np.int32, np.float32] and col not in falha_cols_model + ['id']]
        
        scaler_model = StandardScaler()
        if features_to_scale_model:
            train_df_for_model[features_to_scale_model] = scaler_model.fit_transform(train_df_for_model[features_to_scale_model])
        
        X_train_model = train_df_for_model.drop(columns=['id'] + falha_cols_model)
        y_train_model = train_df_for_model[falha_cols_model]
        trained_feature_columns = list(X_train_model.columns)
        
        final_model_rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced_subsample'))
        final_model_rf.fit(X_train_model, y_train_model)
        
        return final_model_rf, scaler_model, trained_feature_columns, train_medians_model
    except Exception as e:
        st.error(f"Erro crítico ao carregar ou treinar modelo/scaler: {e}")
        return None, None, None, None

# Carregar componentes essenciais
train_df = load_main_data()
model, scaler, feature_columns_for_model, train_medians_for_model = load_model_components()

# --- Navegação na Barra Lateral ---
st.sidebar.title("Menu Principal")
pagina_selecionada = st.sidebar.radio("Navegar para:", ["Página Inicial", "Análise Exploratória (EDA)", "Performace do Modelo", "Simulador Interativo de Defeitos"], index=0)

# --- Conteúdo das Páginas ---
if pagina_selecionada == "Página Inicial":
    st.title("🏭 Análise Preditiva de Defeitos em Chapas de Aço")
    st.markdown("---    ")
    st.subheader("Bem-vindo ao Painel Interativo do Projeto de Controle de Qualidade Industrial")
    st.markdown("""
    Este dashboard é o resultado de um projeto de Ciência de Dados focado na otimização de controle de qualidade na indústria siderúrgica. 
    Utilizando Machine Learning, desenvolvemos um sistema capaz de detectar e classificar múltiplos tipos de defeitos em chapas de aço inoxidável, 
    a partir de dados de inspeção.
    """, unsafe_allow_html=True)
    
    st.markdown("### Objetivos Centrais do Projeto:")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Classificação Inteligente:** Desenvolver um modelo robusto para classificar 7 tipos distintos de defeitos, aumentando a precisão e eficiência da deteção.")
    with col2:
        st.success("**Insights Acionáveis:** Extrair informações valiosas sobre as características dos defeitos e sua relação com as variáveis do processo, fornecendo subsídios para melhorias.")
    
    st.markdown("### Tecnologias e Metodologia:")
    st.markdown("""
    - **Linguagem e Bibliotecas:** Python, Pandas, NumPy, Scikit-learn.
    - **Visualização:** Matplotlib, Seaborn.
    - **Dashboard Interativo:** Streamlit.
    - **Modelo Principal:** Random Forest (Multirrótulo).
    """, unsafe_allow_html=True)
    st.markdown("---    ")
    st.markdown("No menu lateral podemos seguir para uma análise detalhada dos dados, performace do modelo e simulações.")

elif pagina_selecionada == "Análise Exploratória (EDA)":
    st.header("🔎 Análise Exploratória Detalhada dos Dados (EDA)")
    if train_df is not None:
        with st.expander("Visualização do Conjunto de Dados de Treino (Amostra)", expanded=False):
            st.write(f"O conjunto de treino original contém {train_df.shape[0]} registos e {train_df.shape[1]} atributos.")
            st.dataframe(train_df.sample(5)) # Mostrar uma amostra aleatória

        st.subheader("📊 Incidência dos Tipos de Falha")
        falha_cols_eda = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']
        train_df_eda_falhas = train_df.copy()
        map_sim_nao_eda = {'Sim': True, 'sim': True, 'Não': False, 'nao': False, 'Nao': False, True: True, False: False}
        for col in falha_cols_eda:
            if train_df_eda_falhas[col].dtype == 'object':
                train_df_eda_falhas[col] = train_df_eda_falhas[col].map(map_sim_nao_eda)
                train_df_eda_falhas[col] = train_df_eda_falhas[col].fillna(False) # Preenche NaNs resultantes do map
            train_df_eda_falhas[col] = train_df_eda_falhas[col].astype(bool)
        falhas_dist = train_df_eda_falhas[falha_cols_eda].sum().sort_values(ascending=False)
        
        plt.style.use('seaborn-v0_8-colorblind') # Estilo com cores acessíveis
        fig_dist_falhas, ax = plt.subplots(figsize=(12,7))
        bars = sns.barplot(x=falhas_dist.index, y=falhas_dist.values, ax=ax, hue=falhas_dist.index, palette='viridis_r', legend=False)
        ax.set_title('Contagem de Ocorrências por Tipo de Falha', fontsize=18, color='#1a5276', fontweight='bold')
        ax.set_xlabel('Tipo de Falha', fontsize=15, color='#2c3e50')
        ax.set_ylabel('Número de Ocorrências', fontsize=15, color='#2c3e50')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars.patches:
            ax.annotate(f'{int(bar.get_height())}', 
                        (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points', fontsize=11, color='dimgray')
        st.pyplot(fig_dist_falhas)
        st.caption("Análise da frequência de cada tipo de defeito no conjunto de dados de treino. A `falha_5` é a mais prevalente.")

        st.subheader("📉 Distribuição de Características Relevantes")
        if feature_columns_for_model:
            numeric_cols_options = [col for col in train_df.columns if col in feature_columns_for_model and train_df[col].dtype in [np.int64, np.float64]]
            default_ix_hist = numeric_cols_options.index('log_das_areas') if 'log_das_areas' in numeric_cols_options else 0
            feature_to_plot = st.selectbox("Selecione uma característica para visualizar sua distribuição:", 
                                           options=numeric_cols_options,
                                           index=default_ix_hist, key="eda_feature_select")
            if feature_to_plot:
                fig_hist, ax_hist = plt.subplots(figsize=(12,6))
                sns.histplot(train_df[feature_to_plot].dropna(), kde=True, ax=ax_hist, color='#5dade2', edgecolor='#1a5276', bins=30)
                ax_hist.set_title(f"Distribuição da Característica: {feature_to_plot}", fontsize=18, color='#1a5276', fontweight='bold')
                ax_hist.set_xlabel(feature_to_plot, fontsize=15, color='#2c3e50')
                ax_hist.set_ylabel('Frequência', fontsize=15, color='#2c3e50')
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig_hist)
        else:
            st.info("Aguardando carregamento das colunas de features do modelo.")
    else:
        st.warning("Os dados de treino não puderam ser carregados. Verifique o console para erros.")

elif pagina_selecionada == "Performace do Modelo":
    st.header("🏆 Performace do Modelo Preditivo (Random Forest)")
    st.subheader("Principais Métricas de Avaliação (no Conjunto de Validação)")
    results_data = {
        'Métrica': ['Hamming Loss', 'Accuracy (Exact Match Ratio)', 'F1-score (Macro)', 'AUC-ROC (Macro)'],
        'Random Forest': [0.0839, 0.5398, 0.5939, 0.9078],
        'Regressão Logística (Baseline)': [0.1220, 0.4853, 0.5388, 0.8819]
    }
    results_display_df = pd.DataFrame(results_data).set_index('Métrica')
    st.table(results_display_df.style.format("{:.4f}")
             .highlight_max(axis=1, subset=['Random Forest', 'Regressão Logística (Baseline)'], color='#d4efdf', props='font-weight:bold;')
             .set_caption("Comparativo de desempenho entre modelos. Valores mais altos são melhores, exceto para Hamming Loss."))
    st.success("O modelo Random Forest demonstrou um desempenho consistentemente superior, sendo escolhido como o modelo final.")

    st.subheader("🎯 Importância Relativa das Características (Feature Importances)")
    st.markdown("O gráfico a seguir ilustra quais características o modelo Random Forest considerou mais relevantes para realizar as suas previsões. Características com maior importância têm um impacto mais significativo na identificação dos defeitos.")
    try:
        st.image("images/feature_importances_random_forest.png", 
                 caption="Importância Média das Features (Top 15) - Modelo Random Forest Multirrótulo.",
                 use_container_width=True) # Corrigido de use_column_width
    except FileNotFoundError:
        st.warning("Gráfico de importância das features não encontrado. Por favor, execute o script 'feature_importance_analysis.py'.")
    st.caption("Características como `indice_de_variacao_x`, `log_das_areas` e `soma_da_luminosidade` destacam-se como principais preditores.")

elif pagina_selecionada == "Simulador Interativo de Defeitos":
    st.header("⚙️ Simulador Interativo de Previsão de Defeitos")
    st.markdown("Ajuste os valores das características abaixo para simular uma inspeção e obter uma previsão de defeitos do modelo Random Forest.")
    st.info("Obs: Os valores iniciais representam as medianas do conjunto de treino para as características numéricas. As características apresentadas são as utilizadas pelo modelo treinado.")

    if model and scaler and feature_columns_for_model and train_medians_for_model:
        train_df_orig_sim = load_main_data() # Recarregar para garantir que é o original
        if train_df_orig_sim is not None and 'peso_da_placa' in train_df_orig_sim.columns:
            train_df_orig_sim = train_df_orig_sim.drop(columns=['peso_da_placa'])

        input_features = {}
        num_cols_form = 3 # Layout em 3 colunas para os inputs
        form_cols = st.columns(num_cols_form)
        idx = 0
        
        # Ordenar as features para o formulário, talvez pelas mais importantes ou alfabeticamente
        for feature_name in feature_columns_for_model:
            with form_cols[idx % num_cols_form]:
                container = st.container()
                # Obter min, max, median do dataset original para os number_input
                current_median = train_medians_for_model.get(feature_name, 0.0)
                current_min, current_max = 0.0, 100.0 # Valores default
                step_val = 0.1

                if train_df_orig_sim is not None and feature_name in train_df_orig_sim.columns and train_df_orig_sim[feature_name].dtype in [np.float64, np.int64]:
                    current_min = float(train_df_orig_sim[feature_name].min())
                    current_max = float(train_df_orig_sim[feature_name].max())
                    # Ajustar mediana para estar dentro do min/max se necessário
                    current_median = np.clip(float(train_df_orig_sim[feature_name].median()), current_min, current_max)
                    step_val = (current_max - current_min) / 100 if current_max > current_min else 0.01
                    input_features[feature_name] = container.number_input(
                        label=f"{feature_name}", 
                        min_value=current_min, 
                        max_value=current_max, 
                        value=current_median, 
                        step=step_val, 
                        format="%.4f",
                        key=f"sim_{feature_name}"
                    )
                elif feature_name in ['tipo_do_aço_A300', 'tipo_do_aço_A400']:
                    input_features[feature_name] = container.selectbox(
                        f"{feature_name}", 
                        options=[0, 1], 
                        format_func=lambda x: "Sim (Tipo Presente)" if x == 1 else "Não (Tipo Ausente)", 
                        index=0, 
                        key=f"sim_{feature_name}"
                    )
                else: # Fallback para features que não são numéricas ou os tipos de aço (raro se feature_columns_for_model está correto)
                    input_features[feature_name] = container.number_input(label=f"{feature_name}", value=current_median, format="%.4f", key=f"sim_{feature_name}")
            idx += 1
        
        st.markdown("---        ")
        if st.button("Executar Simulação de Previsão", key="run_simulation_button"):
            input_df = pd.DataFrame([input_features])
            input_df_aligned = pd.DataFrame(columns=feature_columns_for_model)
            for col in feature_columns_for_model:
                input_df_aligned[col] = input_df[col] if col in input_df.columns else train_medians_for_model.get(col, 0)
            
            input_processed = preprocess_input_for_prediction(input_df_aligned, scaler, feature_columns_for_model, train_medians_for_model)
            
            if input_processed is not None:
                with st.spinner("A calcular previsões..."):
                    prediction = model.predict(input_processed)
                    prediction_proba = model.predict_proba(input_processed)
                
                st.subheader("Resultado da Simulação:")
                falha_cols_display = [f'Falha {i}' for i in range(1, 7)] + ['Outras Falhas']
                results_pred_list = []
                for i, col_name in enumerate(falha_cols_display):
                    outcome = "DETECTADA  ✅" if prediction[0, i] == 1 else "Não Detectada  ❌"
                    probability = f"{prediction_proba[i][0, 1]*100:.1f}%"
                    results_pred_list.append({"Tipo de Falha": col_name, "Previsão": outcome, "Probabilidade de Ocorrência": probability})
                
                results_pred_final_df = pd.DataFrame(results_pred_list)
                st.table(results_pred_final_df.set_index("Tipo de Falha"))
            else:
                st.error("Não foi possível processar os dados de entrada para a previsão. Verifique os valores.")
    else:
        st.error("O modelo ou os componentes de pré-processamento não foram carregados corretamente. Contacte o suporte.")

st.sidebar.markdown("--- ")
st.sidebar.markdown("**Projeto Bootcamp SENAI**")
st.sidebar.info("Desenvolvido por Renan Mocelin 😊")

