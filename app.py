import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
import os

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Bet Master AI",
    page_icon="⚽",
    layout="centered"
)

st.title("🔮 Bet Master AI")
st.write("---")

# ==============================================================================
# CARREGAMENTO E INTELIGÊNCIA (CACHE)
# ==============================================================================

@st.cache_data
def carregar_dados():
    diretorio = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(diretorio, 'BRA.csv')
    
    try:
        df = pd.read_csv(caminho, sep=';', decimal=',')
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
    st.stop()

@st.cache_resource
def treinar_ia(df):
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

# Função auxiliar Kelly (Reutilizável)
def calcular_kelly(prob_real, odd_site):
    if odd_site <= 1: return 0
    b = odd_site - 1
    q = 1 - prob_real
    f = (b * prob_real - q) / b
    return max(f, 0)

# ==============================================================================
# INTERFACE E LÓGICA
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Configurações")
    metodo_poisson = st.radio("Modelo Matemático", ["Clássico (Multiplicativo)", "Aritmético (Luiz Ramos)"])
    
    st.write("---")
    st.header("💰 Gestão de Banca")
    banca_total = st.number_input("Sua Banca Total (R$)", value=1000.0, step=100.0)
    fracao_kelly = st.slider("Agressividade (Kelly Fracionário)", 0.01, 1.0, 0.10, 0.01)

lista_times = sorted(df['HomeTeam'].unique())

col1, col2 = st.columns(2)
with col1: time_casa = st.selectbox("Time da Casa", lista_times, index=0)
with col2: time_fora = st.selectbox("Time Visitante", lista_times, index=1 if len(lista_times)>1 else 0)

# Inputs de ODDS (Agora globais para usar no Poisson e na IA)
st.write("---")
st.caption("Insira as Odds da Bet365 para calcular o valor da aposta:")
k1, k2, k3 = st.columns(3)
with k1: odd_site_h = st.number_input(f"Odd Vitória {time_casa}", 1.0, 20.0, 2.0, step=0.01)
with k2: odd_site_d = st.number_input(f"Odd Empate", 1.0, 20.0, 3.0, step=0.01)
with k3: odd_site_a = st.number_input(f"Odd Vitória {time_fora}", 1.0, 20.0, 4.0, step=0.01)

if 'calculou' not in st.session_state: st.session_state['calculou'] = False

if st.button("CALCULAR ODDS (POISSON) 🎲", type="primary", use_container_width=True):
    if time_casa == time_fora:
        st.error("Escolha times diferentes!")
    else:
        # Lógica Poisson
        media_gols_casa = df['FTHG'].mean()
        media_gols_fora = df['FTAG'].mean()
        home_stats = df.groupby('HomeTeam')[['FTHG', 'FTAG']].mean()
        away_stats = df.groupby('AwayTeam')[['FTHG', 'FTAG']].mean()
        
        lambda_casa, lambda_fora = 0.0, 0.0

        if metodo_poisson == "Clássico (Multiplicativo)":
            def get_force(time, lado):
                if lado == 'Home':
                    return (home_stats.loc[time, 'FTHG']/media_gols_casa, home_stats.loc[time, 'FTAG']/media_gols_fora) if time in home_stats.index else (1,1)
                else:
                    return (away_stats.loc[time, 'FTAG']/media_gols_fora, away_stats.loc[time, 'FTHG']/media_gols_casa) if time in away_stats.index else (1,1)
            hc_att, hc_def = get_force(time_casa, 'Home')
            aw_att, aw_def = get_force(time_fora, 'Away')
            lambda_casa = hc_att * aw_def * media_gols_casa
            lambda_fora = aw_att * hc_def * media_gols_fora
        else:
            mf_casa = home_stats.loc[time_casa, 'FTHG'] if time_casa in home_stats.index else media_gols_casa
            ms_fora = away_stats.loc[time_fora, 'FTHG'] if time_fora in away_stats.index else media_gols_casa
            lambda_casa = (mf_casa + ms_fora) / 2
            
            mf_fora = away_stats.loc[time_fora, 'FTAG'] if time_fora in away_stats.index else media_gols_fora
            ms_casa = home_stats.loc[time_casa, 'FTAG'] if time_casa in home_stats.index else media_gols_fora
            lambda_fora = (mf_fora + ms_casa) / 2

        prob_h, prob_d, prob_a = 0, 0, 0
        for x in range(7):
            for y in range(7):
                p = poisson.pmf(x, lambda_casa) * poisson.pmf(y, lambda_fora)
                if x > y: prob_h += p
                elif x == y: prob_d += p
                else: prob_a += p
        
        st.session_state.update({
            'calculou': True, 'prob_h': prob_h, 'prob_d': prob_d, 'prob_a': prob_a,
            'l_casa': lambda_casa, 'l_fora': lambda_fora,
            'tc': time_casa, 'tf': time_fora, 'metodo': metodo_poisson
        })

if st.session_state['calculou']:
    tc, tf = st.session_state['tc'], st.session_state['tf']
    ph, pd_prob, pa = st.session_state['prob_h'], st.session_state['prob_d'], st.session_state['prob_a']
    
    st.subheader(f"📊 Probabilidades Históricas ({st.session_state['metodo']})")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Vitória {tc}", f"{ph*100:.1f}%", f"Odd Justa: {1/ph:.2f}")
    c2.metric("Empate", f"{pd_prob*100:.1f}%", f"Odd Justa: {1/pd_prob:.2f}")
    c3.metric(f"Vitória {tf}", f"{pa*100:.1f}%", f"Odd Justa: {1/pa:.2f}")
    
    # Kelly do Poisson
    kh = calcular_kelly(ph, odd_site_h) * fracao_kelly
    kd = calcular_kelly(pd_prob, odd_site_d) * fracao_kelly
    ka = calcular_kelly(pa, odd_site_a) * fracao_kelly
    
    st.caption("💰 Recomendação (Baseada no Histórico):")
    cols_res = st.columns(3)
    if kh > 0: cols_res[0].success(f"R$ {kh*banca_total:.2f}")
    else: cols_res[0].error("Sem Valor")
    if kd > 0: cols_res[1].success(f"R$ {kd*banca_total:.2f}")
    else: cols_res[1].error("Sem Valor")
    if ka > 0: cols_res[2].success(f"R$ {ka*banca_total:.2f}")
    else: cols_res[2].error("Sem Valor")

# ==============================================================================
# IA + KELLY DE MOMENTUM
# ==============================================================================

st.write("---")
with st.expander("🤖 Refinar com Inteligência Artificial (Dados Recentes)", expanded=True):
    st.write("Insira as médias dos últimos 5 jogos.")
    
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
        mapa = {cls: idx for idx, cls in enumerate(classes)}
        
        # Probabilidades da IA
        p_ia_h = probs[mapa['H']]
        p_ia_d = probs[mapa['D']]
        p_ia_a = probs[mapa['A']]
        
        st.markdown("### 🧠 Probabilidades (Momentum/IA)")
        k_ia1, k_ia2, k_ia3 = st.columns(3)
        k_ia1.metric(f"Vitória {time_casa}", f"{p_ia_h*100:.1f}%", f"Odd Justa: {1/p_ia_h:.2f}")
        k_ia2.metric("Empate", f"{p_ia_d*100:.1f}%", f"Odd Justa: {1/p_ia_d:.2f}")
        k_ia3.metric(f"Vitória {time_fora}", f"{p_ia_a*100:.1f}%", f"Odd Justa: {1/p_ia_a:.2f}")

        # KELLY DA IA (Aqui está a sua solicitação!)
        kh_ia = calcular_kelly(p_ia_h, odd_site_h) * fracao_kelly
        kd_ia = calcular_kelly(p_ia_d, odd_site_d) * fracao_kelly
        ka_ia = calcular_kelly(p_ia_a, odd_site_a) * fracao_kelly

        st.caption("💰 Recomendação (Baseada no Momento Atual):")
        cols_ia = st.columns(3)
        
        # Mostra o valor a apostar pela IA
        if kh_ia > 0: cols_ia[0].success(f"APOSTE R$ {kh_ia*banca_total:.2f}")
        else: cols_ia[0].error("Sem Valor")
        
        if kd_ia > 0: cols_ia[1].success(f"APOSTE R$ {kd_ia*banca_total:.2f}")
        else: cols_ia[1].error("Sem Valor")
        
        if ka_ia > 0: cols_ia[2].success(f"APOSTE R$ {ka_ia*banca_total:.2f}")
        else: cols_ia[2].error("Sem Valor")