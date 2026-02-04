import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    # Criar target para gols totais
    df_ml['TotalGoals'] = df_ml['FTHG'] + df_ml['FTAG']
    
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
    
    # Modelo 1: Quem Ganha (Classifier)
    y_winner = df_ml['FTR']
    modelo_winner = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_winner.fit(X, y_winner)
    
    # Modelo 2: Quantos Gols (Regressor)
    y_goals = df_ml['TotalGoals']
    modelo_goals = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_goals.fit(X, y_goals)
    
    return modelo_winner, modelo_goals, features

modelo_winner, modelo_goals, features_ia = treinar_ia(df)

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

        # CÁLCULO DE PROBABILIDADES COMPLETO (Incluindo BTTS e 3.5)
        prob_h, prob_d, prob_a = 0, 0, 0
        prob_over_15, prob_over_25, prob_over_35 = 0, 0, 0
        prob_ambas_marcam = 0
        prob_gols_exatos = {0:0, 1:0, 2:0, 3:0, 4:0}
        
        for x in range(7):
            for y in range(7):
                p = poisson.pmf(x, lambda_casa) * poisson.pmf(y, lambda_fora)
                
                # Resultado 1x2
                if x > y: prob_h += p
                elif x == y: prob_d += p
                else: prob_a += p
                
                # Over/Under
                total_gols = x + y
                if total_gols > 1.5: prob_over_15 += p
                if total_gols > 2.5: prob_over_25 += p
                if total_gols > 3.5: prob_over_35 += p
                
                # Ambas Marcam (BTTS)
                if x > 0 and y > 0: prob_ambas_marcam += p
                
                # Gols Exatos
                if total_gols >= 4: prob_gols_exatos[4] += p
                else: prob_gols_exatos[total_gols] += p
        
        st.session_state.update({
            'calculou': True, 'prob_h': prob_h, 'prob_d': prob_d, 'prob_a': prob_a,
            'l_casa': lambda_casa, 'l_fora': lambda_fora,
            'p_over15': prob_over_15, 'p_over25': prob_over_25, 'p_over35': prob_over_35,
            'p_btts': prob_ambas_marcam, 'p_gols_exatos': prob_gols_exatos,
            'tc': time_casa, 'tf': time_fora, 'metodo': metodo_poisson
        })

# --- EXIBIÇÃO ---
odd_site_h, odd_site_d, odd_site_a = 0.0, 0.0, 0.0
odd_site_o15, odd_site_o25 = 0.0, 0.0

if st.session_state['calculou']:
    tc, tf = st.session_state['tc'], st.session_state['tf']
    ph, pd_prob, pa = st.session_state['prob_h'], st.session_state['prob_d'], st.session_state['prob_a']
    
    st.subheader(f"📊 Probabilidades Históricas ({st.session_state['metodo']})")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Vitória {tc}", f"{ph*100:.1f}%", f"Odd Justa: {1/ph:.2f}")
    c2.metric("Empate", f"{pd_prob*100:.1f}%", f"Odd Justa: {1/pd_prob:.2f}")
    c3.metric(f"Vitória {tf}", f"{pa*100:.1f}%", f"Odd Justa: {1/pa:.2f}")
    
    # --- ÁREA DE GOLS (RESTAURADA E MELHORADA) ---
    st.write("---")
    st.subheader("⚽ Mercado de Gols (Over/Under)")
    
    # Abas como você gosta
    tab1, tab2 = st.tabs(["Over / Under", "Gols Exatos & BTTS"])
    
    with tab1:
        mg1, mg2, mg3 = st.columns(3)
        mg1.metric("Over 1.5 Gols", f"{st.session_state['p_over15']*100:.1f}%", f"Odd: {1/st.session_state['p_over15']:.2f}")
        mg2.metric("Over 2.5 Gols", f"{st.session_state['p_over25']*100:.1f}%", f"Odd: {1/st.session_state['p_over25']:.2f}")
        mg3.metric("Over 3.5 Gols", f"{st.session_state['p_over35']*100:.1f}%", f"Odd: {1/st.session_state['p_over35']:.2f}")
        st.caption("Over 1.5 = + de 1 gol no jogo | Over 2.5 = + de 2 gols no jogo.")

    with tab2:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("##### 🥅 Gols Totais no Jogo")
            p_ex = st.session_state['p_gols_exatos']
            st.write(f"0 Gols: **{p_ex[0]*100:.1f}%** (Odd {1/p_ex[0]:.2f})")
            st.write(f"1 Gol: **{p_ex[1]*100:.1f}%** (Odd {1/p_ex[1]:.2f})")
            st.write(f"2 Gols: **{p_ex[2]*100:.1f}%** (Odd {1/p_ex[2]:.2f})")
            st.write(f"3 Gols: **{p_ex[3]*100:.1f}%** (Odd {1/p_ex[3]:.2f})")
            st.write(f"4+ Gols: **{p_ex[4]*100:.1f}%** (Odd {1/p_ex[4]:.2f})")
        with col_b2:
            st.markdown("##### 🤝 Ambas Marcam (BTTS)")