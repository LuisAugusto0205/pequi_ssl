Este repostório utiliza a biblioteca [rsoccer](https://github.com/robocin/rSoccer) para aplicar algortimos de Aprendizado por Reforço no ambinete Small Size League (SSL).

# Configurando o ambiente

Clone o reposítorio com o comando:

    git clone https://github.com/LuisAugusto0205/pequi_ssl.git

Após isso, crie um ambiente virtual com virtualenv:

    virtualenv nome_ambiente

Então, ative o ambiente com:
    
    source nome_ambiente/bin/activate


Agora é necessário instalar a dependências do projeto, para isso primeiro rode o commando na pasta raiz do projeto:

    pip install .

Em seguida, instale outros pacotes necessários com:

    pip install -r requirements.txt

# Rodar Algoritmo de Reforço

Até o momento está sendo realizado testes com a implementação do algoritmo DDPG do respositório [cleanrl](https://github.com/vwxyzjn/cleanrl)

Para rodar o DDPG, execute:

    python teste_cleanrl_ddpg.py

# Rodar controlando manualmente

Para controlar o agente manulamente é necessário instalar a biblioteca abrir dois terminais diferentes. No primeiro rode o arquivo `simulation.py` com o commando:

    python simulation.py

Depois, no segundo terminal, execute o arquivo `control.py` com:

    python control.py

Pronto! basta deixar o segundo terminal selecionado e controlar o robô com as teclas `a, w, s, d` para movimentar e `q, e` para rotacionar.