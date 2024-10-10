Este repostório utiliza a biblioteca [rsoccer](https://github.com/robocin/rSoccer) para aplicar algortimos de Aprendizado por Reforço no ambinete Small Size League (SSL).

# Configurando o ambiente

Clone o reposítorio com o comando:

    git clone https://github.com/LuisAugusto0205/pequi_ssl.git

Mude para branch rllib:

    git checkout rllib

Construa a imagem:

    docker build -t rsoccer .

Rode o container com volume:

    docker run --gpus all --name rsoccer -v $(pwd)/volume:/app/volume -it rsoccer python rllib_multiagent.py --task 1 --checkpoint_task 1

A variável `task` indica qual task será executada, podendo ser:
    1: Posição do robô fixa e bola fixa
    2: Posição do robô aleatória e bola fixa.
    3: Posição do robê aleatória e bola em posição aleatória:

O `checkpoint_task` indica o checkpoint de qual task deve ser carrega para iniciar o aprendizado por reforço. 

Caso não esteja reconhecendo a gpu, tente instalar o [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) ou mudar a versão do cuda no dockerfile

