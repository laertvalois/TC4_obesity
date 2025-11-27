import pickle

try:
    with open('model/obesity_model.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("✅ Modelo carregado com sucesso!")
    print(f"   Modelo: {data.get('model_name', 'N/A')}")
    print(f"   Acurácia: {data.get('accuracy', 'N/A')}")
    print(f"   Features: {len(data.get('columns', []))}")
    print(f"   Encoders: {list(data.get('label_encoders', {}).keys())}")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")

