"""Script de teste rápido para verificar se a aplicação funciona"""
import sys
import os

# Adicionar o diretório atual ao path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Testar importações
    import streamlit as st
    import pandas as pd
    import pickle
    import numpy as np
    from PIL import Image
    
    print("✅ Todas as importações OK")
    
    # Testar carregamento do modelo
    with open('model/obesity_model.pkl', 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    scaler = data['scaler']
    label_encoders = data['label_encoders']
    columns = data['columns']
    
    print("✅ Modelo carregado com sucesso")
    print(f"   - Modelo: {data.get('model_name', 'N/A')}")
    print(f"   - Acurácia: {data.get('accuracy', 'N/A'):.2%}")
    print(f"   - Número de features: {len(columns)}")
    
    # Testar carregamento do dataset
    if os.path.exists('data/Obesity.csv'):
        df = pd.read_csv('data/Obesity.csv')
        print(f"✅ Dataset carregado: {len(df)} registros")
    else:
        print("⚠️  Dataset não encontrado (não crítico para o app)")
    
    # Verificar gráficos
    graphs_dir = 'graphs'
    if os.path.exists(graphs_dir):
        graphs = [f for f in os.listdir(graphs_dir) if f.endswith('.png')]
        print(f"✅ Gráficos encontrados: {len(graphs)}")
        for g in graphs:
            print(f"   - {g}")
    else:
        print("⚠️  Diretório de gráficos não encontrado")
    
    print("\n🎉 Aplicação pronta para executar!")
    print("   Execute: streamlit run app.py")
    print("   A aplicação abrirá em: http://localhost:8501")
    
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()

