# Use uma imagem base Python oficial
FROM python:3.10-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo requirements.txt para o contêiner
COPY requirements.txt .

# Instale as dependências do Python listadas em requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

# Copie a pasta DatabaseAtividades para o contêiner
RUN mkdir /rsoccer_gym
COPY rsoccer_gym rsoccer_gym
COPY rllib_multiagent.py .

# Comando para rodar o Streamlit na inicialização do contêiner
CMD ["python", "rllib_multiagent.py", "--checkpoint=volume/last_checkpoint", "--logdir=volume/log_tensor/gotoball"]
