# 🚀 Guia de Deploy - Sistema Preditivo de Obesidade

Este guia contém instruções detalhadas para fazer o deploy do sistema preditivo de obesidade no Streamlit Cloud.

## 📋 Pré-requisitos

1. Conta no [Streamlit Cloud](https://streamlit.io/cloud)
2. Repositório no GitHub com todo o código
3. Arquivo `requirements.txt` atualizado
4. Modelo treinado salvo em `model/obesity_model.pkl`

## 🔧 Passo a Passo

### 1. Preparar o Repositório GitHub

```bash
# Certifique-se de que todos os arquivos estão commitados
git add .
git commit -m "Preparação para deploy"
git push origin main
```

### 2. Estrutura de Arquivos Necessária

Certifique-se de que seu repositório tenha a seguinte estrutura:

```
TC4_obesity/
├── app.py                    # Aplicação Streamlit
├── train_model.py            # Script de treinamento
├── requirements.txt          # Dependências
├── README.md                 # Documentação
├── data/
│   └── Obesity.csv          # Dataset (opcional no deploy)
├── model/
│   └── obesity_model.pkl   # Modelo treinado (OBRIGATÓRIO)
└── graphs/                  # Gráficos (opcional, serão gerados)
```

### 3. Deploy no Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Faça login com sua conta GitHub
3. Clique em "New app"
4. Selecione:
   - **Repository**: Seu repositório
   - **Branch**: main (ou master)
   - **Main file path**: `app.py`
5. Clique em "Deploy"

### 4. Configurações Adicionais (Opcional)

Crie um arquivo `.streamlit/config.toml` na raiz do projeto:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

### 5. Verificar Deploy

Após o deploy, você receberá um link no formato:
```
https://seu-usuario-streamlit-app.streamlit.app
```

Teste todas as funcionalidades:
- ✅ Página de previsão
- ✅ Dashboard analítico
- ✅ Visualizações de gráficos
- ✅ Cálculo de IMC

## 📊 Dashboard Analítico Separado (Opcional)

Se quiser criar um dashboard analítico separado, crie um arquivo `dashboard.py`:

```python
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Analítico", layout="wide")
# ... código do dashboard
```

E faça deploy como um segundo app no Streamlit Cloud.

## 🔗 Links para Entrega

Após o deploy, você terá:

1. **Link da Aplicação Preditiva**: `https://seu-usuario-streamlit-app.streamlit.app`
2. **Link do Dashboard Analítico**: (mesmo link, página separada no app)
3. **Link do Repositório GitHub**: `https://github.com/seu-usuario/TC4_obesity`

## ⚠️ Troubleshooting

### Erro: "Module not found"
- Verifique se todas as dependências estão em `requirements.txt`
- Certifique-se de que as versões são compatíveis

### Erro: "File not found"
- Verifique os caminhos dos arquivos (use caminhos relativos)
- Certifique-se de que `model/obesity_model.pkl` existe

### Erro: "Model not loading"
- Execute `train_model.py` localmente primeiro
- Commit o arquivo `model/obesity_model.pkl` no repositório

### App lento
- Otimize o carregamento de dados
- Use cache do Streamlit: `@st.cache_data`

## 📝 Checklist de Deploy

- [ ] Código commitado no GitHub
- [ ] `requirements.txt` atualizado
- [ ] Modelo treinado (`model/obesity_model.pkl`) commitado
- [ ] Gráficos gerados (opcional, podem ser gerados no deploy)
- [ ] App deployado no Streamlit Cloud
- [ ] Todos os links funcionando
- [ ] Testes realizados em produção

## 🎥 Vídeo de Apresentação

Lembre-se de gravar um vídeo (4-10 min) mostrando:
- Estratégia utilizada
- Pipeline de ML
- Sistema preditivo em funcionamento
- Dashboard analítico
- Insights para equipe médica

Boa sorte! 🚀

