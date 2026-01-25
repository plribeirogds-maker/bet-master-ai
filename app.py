import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
import os

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA (Título, Ícone, Layout)
# ==============================================================================
st.set_page_config(
    page_title="Bet Master AI",
    page_icon="⚽",
    layout="centered"
)

# Título Bonito e Estilizado
st.title("🔮 Bet Master AI")
st.write("---")

# ==============================================================================
# CARREGAMENTO E INTELIGÊNCIA (CACHE)
# ==============================================================================

# O @st.cache_data faz com que o Python leia o arquivo SÓ UMA VEZ.
# O app fica super rápido.
@st.cache_data
def carregar_dados():
    diretorio = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(diretorio, 'BRA.csv')
    
    try:
        df = pd.read_csv(caminho, sep=';', decimal=',')
        # Limpezas
        df = df.dropna(subset=['HG', 'AG', 'Res', 'Home', 'Away'])
        df = df.rename(columns={
            'HG': 'FTHG', 'AG': 'FTAG', 
            'Home': 'HomeTeam', 'Away': 'AwayTeam', 
            'Res': 'FTR'
        })
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')
        return df
    except FileNotFoundError:
        return None

df = carregar_dados()

if df is None:
    st.error("ERRO: O arquivo 'BRA.csv' não foi encontrado na pasta!")
    st.stop() # Para o código aqui se não tiver arquivo

# Treinamento da IA (Também em cache para não treinar toda hora)
@st.cache_resource
def treinar_ia(df):
    # Prepara dados (feature engineering igual fizemos antes)
    df_ml = df.copy()
    df_ml['H_Pts'] = np.where(df_ml['FTR'] == 'H', 3, np.where(df_ml['FTR'] == 'D', 1, 0))
    df_ml['A_Pts'] = np.where(df_ml['FTR'] == 'A', 3, np.where(df_ml['FTR'] == 'D', 1, 0))
    
    cols_home = ['Date', 'HomeTeam', 'FTHG', 'FTAG', 'H_Pts']
    cols_away = ['Date', 'AwayTeam', 'FTAG', 'FTHG', 'A_Pts']
    
    h_games = df_ml[cols_home].rename(columns={'HomeTeam':'Team', 'FTHG':'GS', 'FTAG':'GC', 'H_Pts':'Pts'})
    a_games = df_ml[cols_away].rename(columns={'AwayTeam':'Team', 'FTAG':'GS', 'FTHG':'GC', 'A_Pts':'Pts'})
    
    all_games = pd.concat([h_games, a_games]).sort_values(['Team', 'Date'])
    
    grouped = all_games.groupby('Team')
    all_games['L5_Pts'] = grouped['Pts'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    all_games['L5_GS'] = grouped['GS'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    all_games['L5_GC'] = grouped['GC'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    df_ml = pd.merge(df_ml, all_games[['Date', 'Team', 'L5_Pts', 'L5_GS', 'L5_GC']], 
                     left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    df_ml = df_ml.rename(columns={'L5_Pts': 'H_L5_Pts', 'L5_GS': 'H_L5_GS', 'L5_GC': 'H_L5_GC'}).drop(columns=['Team'])
    
    df_ml = pd.merge(df_ml, all_games[['Date', 'Team', 'L5_Pts', 'L5_GS', 'L5_GC']], 
                     left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    df_ml = df_ml.rename(columns={'L5_Pts': 'A_L5_Pts', 'L5_GS': 'A_L5_GS', 'L5_GC': 'A_L5_GC'}).drop(columns=['Team'])
    
    df_ml = df_ml.dropna()
    
    features = ['H_L5_Pts', 'H_L5_GS', 'H_L5_GC', 'A_L5_Pts', 'A_L5_GS', 'A_L5_GC']
    X = df_ml[features]
    y = df_ml['FTR']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    return modelo, features

modelo_ia, features_ia = treinar_ia(df)

# ==============================================================================
# INTERFACE DO USUÁRIO (O APP)
# ==============================================================================

# Barra lateral para informações
with st.sidebar:
    st.header("⚙️ Configurações")
    st.info(f"Base de Dados: {len(df)} jogos carregados.")
    st.markdown("Desenvolvido com Python & Streamlit")

# Seleção de Times (DROPDOWN - Muito melhor que digitar!)
lista_times = sorted(df['HomeTeam'].unique())

col1, col2 = st.columns(2)
with col1:
    time_casa = st.selectbox("Time da Casa (Mandante)", lista_times, index=0)
with col2:
    # Tenta selecionar um rival diferente automaticamente para não ficar igual
    index_fora = 1 if len(lista_times) > 1 else 0
    time_fora = st.selectbox("Time Visitante", lista_times, index=index_fora)

# Botão Principal
if st.button("CALCULAR ODDS 🎲", type="primary", use_container_width=True):
    
    if time_casa == time_fora:
        st.error("Escolha times diferentes!")
    else:
        # --- LÓGICA POISSON ---
        media_gols_casa = df['FTHG'].mean()
        media_gols_fora = df['FTAG'].mean()
        
        home_stats = df.groupby('HomeTeam')[['FTHG', 'FTAG']].mean()
        away_stats = df.groupby('AwayTeam')[['FTHG', 'FTAG']].mean()

        def get_force(time, lado):
            if lado == 'Home':
                stats = home_stats
                if time in stats.index:
                    att = stats.loc[time, 'FTHG'] / media_gols_casa
                    defi = stats.loc[time, 'FTAG'] / media_gols_fora
                    return att, defi
            else:
                stats = away_stats
                if time in stats.index:
                    att = stats.loc[time, 'FTAG'] / media_gols_fora
                    defi = stats.loc[time, 'FTHG'] / media_gols_casa
                    return att, defi
            return 1.0, 1.0

        hc_att, hc_def = get_force(time_casa, 'Home')
        aw_att, aw_def = get_force(time_fora, 'Away')

        lambda_casa = hc_att * aw_def * media_gols_casa
        lambda_fora = aw_att * hc_def * media_gols_fora

        prob_h, prob_d, prob_a = 0, 0, 0
        for x in range(7):
            for y in range(7):
                p = poisson.pmf(x, lambda_casa) * poisson.pmf(y, lambda_fora)
                if x > y: prob_h += p
                elif x == y: prob_d += p
                else: prob_a += p

        # --- EXIBIÇÃO DOS RESULTADOS ---
        st.subheader("📊 Análise Estatística (Histórico)")
        
        # Colunas de Probabilidade (Métricas Grandes)
        c1, c2, c3 = st.columns(3)
        c1.metric(label=f"Vitória {time_casa}", value=f"{prob_h*100:.1f}%", delta=f"Odd: {1/prob_h:.2f}")
        c2.metric(label="Empate", value=f"{prob_d*100:.1f}%", delta=f"Odd: {1/prob_d:.2f}")
        c3.metric(label=f"Vitória {time_fora}", value=f"{prob_a*100:.1f}%", delta=f"Odd: {1/prob_a:.2f}")

        # Gráfico de Barras
        chart_data = pd.DataFrame({
            "Resultado": [time_casa, "Empate", time_fora],
            "Probabilidade": [prob_h, prob_d, prob_a]
        })
        st.bar_chart(chart_data, x="Resultado", y="Probabilidade", color=["#1f77b4"])
        
        st.caption(f"Placar Esperado: {time_casa} {lambda_casa:.2f} x {lambda_fora:.2f} {time_fora}")

# ==============================================================================
# ÁREA DA INTELIGÊNCIA ARTIFICIAL (Expander para não poluir)
# ==============================================================================

st.write("---")
with st.expander("🤖 Refinar com Inteligência Artificial (Dados Recentes)"):
    st.write("Insira as médias dos últimos 5 jogos (Geral) para ver a opinião da IA.")
    
    col_ia1, col_ia2 = st.columns(2)
    
    with col_ia1:
        st.markdown(f"**{time_casa}**")
        hp = st.number_input("Pontos (Média)", 0.0, 3.0, 1.5, step=0.1, key='hp')
        hgs = st.number_input("Gols Feitos (Média)", 0.0, 5.0, 1.2, step=0.1, key='hgs')
        hgc = st.number_input("Gols Sofridos (Média)", 0.0, 5.0, 1.0, step=0.1, key='hgc')

    with col_ia2:
        st.markdown(f"**{time_fora}**")
        ap = st.number_input("Pontos (Média)", 0.0, 3.0, 1.5, step=0.1, key='ap')
        ags = st.number_input("Gols Feitos (Média)", 0.0, 5.0, 1.2, step=0.1, key='ags')
        agc = st.number_input("Gols Sofridos (Média)", 0.0, 5.0, 1.0, step=0.1, key='agc')
        
    if st.button("Consultar o Robô 🤖"):
        input_data = pd.DataFrame([[hp, hgs, hgc, ap, ags, agc]], columns=features_ia)
        probs = modelo_ia.predict_proba(input_data)[0]
        classes = modelo_ia.classes_
        
        # Mapeamento seguro das classes
        mapa = {cls: idx for idx, cls in enumerate(classes)}
        
        p_casa = probs[mapa['H']]
        p_emp = probs[mapa['D']]
        p_fora = probs[mapa['A']]
        
        st.success(f"🧠 Previsão da IA:")
        
        # Mostra quem é o favorito da IA com destaque
        if p_casa > p_fora and p_casa > p_emp:
            st.write(f"O modelo aponta favoritismo para o **{time_casa}** ({p_casa*100:.1f}%)")
        elif p_fora > p_casa and p_fora > p_emp:
            st.write(f"O modelo aponta favoritismo para o **{time_fora}** ({p_fora*100:.1f}%)")
        else:
            st.warning(f"O modelo prevê um jogo muito equilibrado/empate.")
            
        st.progress(int(p_casa*100), text=f"Força {time_casa}")
        st.progress(int(p_fora*100), text=f"Força {time_fora}")