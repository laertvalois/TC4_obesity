import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# --- Carregar modelo e artefatos ---
with open('model/obesity_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
label_encoders = data['label_encoders']
columns = data['columns']

# --- Sidebar de navegação ---
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Previsão de Obesidade", "Insights e Métricas"])

# --- Página 1: Previsão ---
if page == "Previsão de Obesidade":
    st.title("🏥 Preditor de Nível de Obesidade")
    st.markdown("Responda as perguntas abaixo para estimar o nível de obesidade:")

    # Perguntas categóricas
    user_input = {}
    user_input["Gender"] = st.selectbox("Gênero:", ["Male", "Female"])
    user_input["Age"] = st.slider("Idade (anos):", 10, 80, 25)
    user_input["Height"] = st.number_input("Altura (m):", min_value=1.20, max_value=2.10, value=1.70, step=0.01)
    user_input["Weight"] = st.number_input("Peso (kg):", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    user_input["family_history"] = st.selectbox("Algum membro da família sofre ou sofreu de obesidade?", ["no", "yes"])

    st.subheader("Hábitos alimentares")
    user_input["FAVC"] = st.selectbox("Você come alimentos altamente calóricos com frequência?", ["no", "yes"])
    user_input["FCVC"] = st.slider("Você costuma comer vegetais nas refeições? (1=nunca, 3=sempre)", 1, 3, 2)
    user_input["NCP"] = st.slider("Quantas refeições principais você faz por dia?", 1, 4, 3)
    user_input["CAEC"] = st.selectbox("Você come algo entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    user_input["SMOKE"] = st.selectbox("Você fuma?", ["no", "yes"])

    st.subheader("Hábitos diários")
    user_input["CH2O"] = st.slider("Quanta água você bebe por dia? (1=pouca, 3=muita)", 1, 3, 2)
    user_input["SCC"] = st.selectbox("Você monitora as calorias que ingere?", ["no", "yes"])
    user_input["FAF"] = st.slider("Com que frequência pratica atividade física? (0=nunca, 3=frequente)", 0, 3, 2)
    user_input["TUE"] = st.slider("Tempo de uso de dispositivos eletrônicos (0=baixo, 2=alto)", 0, 2, 1)
    
    # Calcular IMC automaticamente
    bmi = user_input["Weight"] / (user_input["Height"] ** 2)
    st.info(f"📊 **IMC Calculado:** {bmi:.2f} kg/m²")
    user_input["CALC"] = st.selectbox("Com que frequência você bebe álcool?", ["no", "Sometimes", "Frequently", "Always"])
    user_input["MTRANS"] = st.selectbox("Meio de transporte principal:", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

    # Prever
    if st.button("Classificar"):
        df_input = pd.DataFrame([user_input])
        
        # --- Feature Engineering (igual ao treinamento) ---
        # 1. Criar IMC
        df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)
        
        # 2. Criar categoria de IMC
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            elif bmi < 35:
                return 'Obese_I'
            elif bmi < 40:
                return 'Obese_II'
            else:
                return 'Obese_III'
        
        df_input['BMI_Category'] = df_input['BMI'].apply(categorize_bmi)
        
        # 3. Codificar todas as variáveis categóricas (exceto Obesity)
        for col, le in label_encoders.items():
            if col in df_input.columns and col != 'Obesity':
                try:
                    df_input[col] = le.transform(df_input[col].astype(str))
                except:
                    # Se valor não estiver no encoder, usar o primeiro valor
                    df_input[col] = 0
        
        # 4. Criar Risk Score após codificação
        # Encontrar índices de 'yes' nos encoders
        try:
            favc_le = label_encoders.get('FAVC')
            family_le = label_encoders.get('family_history')
            
            if favc_le is not None and family_le is not None:
                # Encontrar índice de 'yes' em cada encoder
                yes_favc_idx = None
                yes_family_idx = None
                
                for i, val in enumerate(favc_le.classes_):
                    if str(val).lower() == 'yes':
                        yes_favc_idx = i
                        break
                
                for i, val in enumerate(family_le.classes_):
                    if str(val).lower() == 'yes':
                        yes_family_idx = i
                        break
                
                # Se não encontrar 'yes', usar 1 como padrão
                if yes_favc_idx is None:
                    yes_favc_idx = 1 if len(favc_le.classes_) > 1 else 0
                if yes_family_idx is None:
                    yes_family_idx = 1 if len(family_le.classes_) > 1 else 0
                
                # Criar Risk Score
                df_input['Risk_Score'] = (
                    (df_input['FAVC'] == yes_favc_idx).astype(int) +
                    (df_input['family_history'] == yes_family_idx).astype(int) -
                    (df_input['FAF'] / 3.0) +
                    (df_input['TUE'] / 2.0)
                )
            else:
                # Fallback simples
                df_input['Risk_Score'] = df_input['FAVC'] + df_input['family_history'] - (df_input['FAF'] / 3.0) + (df_input['TUE'] / 2.0)
        except Exception as e:
            # Fallback em caso de erro
            df_input['Risk_Score'] = df_input.get('FAVC', 0) + df_input.get('family_history', 0) - (df_input.get('FAF', 0) / 3.0) + (df_input.get('TUE', 0) / 2.0)
        
        # 5. Garantir que todas as colunas esperadas estejam presentes e na ordem correta
        expected_cols = columns if isinstance(columns, list) else list(columns)
        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = 0  # Valor padrão para colunas faltantes
        
        # Reordenar colunas na ordem esperada pelo modelo
        df_input = df_input[expected_cols]
        
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)[0]
        inv_pred = list(label_encoders["Obesity"].inverse_transform([pred]))[0]

        # Calcular probabilidades
        proba = model.predict_proba(df_scaled)[0]
        proba_dict = dict(zip(label_encoders["Obesity"].classes_, proba))
        sorted_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
        
        st.success(f"🏷️ **Nível de obesidade previsto: {inv_pred}**")
        
        st.subheader("📊 Probabilidades por Classe:")
        for classe, prob in sorted_proba[:3]:  # Top 3
            bar_color = "🟢" if prob < 0.3 else "🟡" if prob < 0.6 else "🔴"
            st.progress(prob, text=f"{bar_color} {classe}: {prob:.1%}")

# --- Página 2: Dashboard Analítico ---
elif page == "Insights e Métricas":
    st.title("📊 Dashboard Analítico - Previsão de Obesidade")
    st.markdown("### Visão estratégica para equipe médica")
    
    # Carregar dados para análise
    try:
        df = pd.read_csv('data/Obesity.csv')
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Pacientes", len(df))
        with col2:
            st.metric("Taxa de Obesidade", f"{(df['Obesity'].str.contains('Obesity', case=False).sum() / len(df) * 100):.1f}%")
        with col3:
            avg_age = df['Age'].mean()
            st.metric("Idade Média", f"{avg_age:.1f} anos")
        with col4:
            avg_bmi = (df['Weight'] / (df['Height'] ** 2)).mean()
            st.metric("IMC Médio", f"{avg_bmi:.1f} kg/m²")
        
        st.markdown("---")
        
        # Seção 1: Desempenho do Modelo
        st.header("🎯 Desempenho do Modelo Preditivo")
        
        try:
            with open('model/metrics.txt', 'r', encoding='utf-8') as f:
                metrics_text = f.read()
            st.text_area("Métricas Detalhadas", metrics_text, height=200)
        except:
            st.info("Execute train_model.py para gerar as métricas")
        
        st.markdown("### 🔹 Comparação de Acurácia entre Modelos")
        try:
            img_comp = Image.open("graphs/model_comparison.png")
            st.image(img_comp, caption="Comparação de Acurácia entre os Modelos", width='stretch')
        except:
            st.warning("Imagem não encontrada. Execute train_model.py primeiro.")

        st.markdown("### 🔹 Matriz de Confusão")
        try:
            img_conf = Image.open("graphs/confusion_matrix.png")
            st.image(img_conf, caption="Matriz de Confusão do Melhor Modelo", width='stretch')
        except:
            st.warning("Matriz de confusão não encontrada.")

        st.markdown("### 🔹 Importância das Features")
        try:
            img_feat = Image.open("graphs/feature_importance.png")
            st.image(img_feat, caption="Top 15 Features Mais Importantes", width='stretch')
        except:
            st.warning("Gráfico de importância não encontrado.")

        st.markdown("---")
        
        # Seção 2: Análise Exploratória
        st.header("📈 Análise Exploratória dos Dados")
        
        st.markdown("### 🔹 Distribuição das Classes de Obesidade")
        try:
            img_dist = Image.open("graphs/target_distribution.png")
            st.image(img_dist, caption="Distribuição das Classes", width='stretch')
        except:
            # Criar gráfico inline se não existir
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Obesity'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title("Distribuição das Classes de Obesidade")
            ax.set_xlabel("Nível de Obesidade")
            ax.set_ylabel("Frequência")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("### 🔹 Correlação entre Variáveis")
        try:
            img_corr = Image.open("graphs/correlation_heatmap.png")
            st.image(img_corr, caption="Mapa de Correlação entre Variáveis", width='stretch')
        except:
            st.warning("Mapa de correlação não encontrado.")
        
        st.markdown("---")
        
        # Seção 3: Insights para Equipe Médica
        st.header("💡 Insights Estratégicos para Equipe Médica")
        
        # Análises específicas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Fatores de Risco Identificados")
            st.markdown("""
            - **Histórico Familiar**: Pacientes com histórico familiar têm maior risco
            - **Alimentos Calóricos (FAVC)**: Consumo frequente aumenta significativamente o risco
            - **Sedentarismo**: Baixa atividade física (FAF) está correlacionada com obesidade
            - **Tempo em Dispositivos (TUE)**: Maior tempo de uso aumenta o risco
            - **Poucas Refeições (NCP)**: Menos refeições principais pode indicar padrões não saudáveis
            """)
        
        with col2:
            st.subheader("✅ Fatores Protetores")
            st.markdown("""
            - **Atividade Física Regular (FAF)**: Reduz significativamente o risco
            - **Consumo de Vegetais (FCVC)**: Hábito protetor importante
            - **Monitoramento Calórico (SCC)**: Consciência alimentar ajuda na prevenção
            - **Hidratação Adequada (CH2O)**: Importante para metabolismo
            - **Transporte Ativo**: Caminhar ou usar bicicleta reduz risco
            """)
        
        st.markdown("---")
        
        # Análises interativas
        st.subheader("📊 Análises Interativas")
        
        # Análise por gênero
        st.markdown("#### Distribuição por Gênero")
        gender_obesity = pd.crosstab(df['Gender'], df['Obesity'], normalize='index') * 100
        st.bar_chart(gender_obesity)
        
        # Análise por idade
        st.markdown("#### Relação Idade vs Obesidade")
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 100], labels=['<20', '20-30', '30-40', '40-50', '50+'])
        age_bmi = df.groupby('Age_Group', observed=True)['BMI'].mean()
        st.line_chart(age_bmi)
        
        # Análise de atividade física
        st.markdown("#### Impacto da Atividade Física")
        activity_obesity = pd.crosstab(df['FAF'], df['Obesity'].str.contains('Obesity', case=False), normalize='index') * 100
        st.bar_chart(activity_obesity)
        
        st.markdown("---")
        
        # Recomendações
        st.header("🎯 Recomendações Clínicas")
        st.markdown("""
        ### Para Prevenção e Tratamento:
        
        1. **Triagem Familiar**: Priorizar pacientes com histórico familiar de obesidade
        2. **Educação Alimentar**: Focar em redução de alimentos altamente calóricos
        3. **Promoção de Atividade Física**: Incentivar exercícios regulares
        4. **Monitoramento de IMC**: Acompanhamento regular para detecção precoce
        5. **Redução de Tempo em Dispositivos**: Orientar sobre tempo de tela
        6. **Padrões Alimentares**: Encorajar refeições regulares e balanceadas
        
        ### Uso do Modelo Preditivo:
        - O modelo pode auxiliar na **identificação precoce** de risco
        - Use como **ferramenta complementar** ao diagnóstico clínico
        - Considere os fatores de risco identificados no **aconselhamento** ao paciente
        - **Validação clínica** sempre necessária para decisões de tratamento
        """)
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.info("Certifique-se de que o arquivo data/Obesity.csv existe")
