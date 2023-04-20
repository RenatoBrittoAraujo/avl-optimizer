## Descrição de projeto

Download do AVL (linux, windows, mac): https://web.mit.edu/drela/Public/web/avl/

REQUISITOS:
a idéia era fazer um algoritmo de otimização em python que faz interface com um outro programa.chamado.AVL
o AVL é programa de linha de comando
a matemática eu já sei a lógica
só essa parte de interface, a otimização em si ainda não
windows né? sim
a idéia é basicamente rodar uma sequência de simulações no AVL
usar umas derivadas pra ir melhorando uma dada superfície aerodinâmica
tem um .txt que ele lê pra formar as superfícies
linha de comando é tipo isso aqui, o terminal, pensei que era assim que o programa era chamado. mas é pela janela mesmo
seria primeiro load pra botar o arquivo que o python gerar
depois oper
x, pq provavelmente não vai precisar mudar manualmente os valores
no máximo load do run case
depois st, o nome do arquivo de saída
aí ler o arquivo de saída, gerar uma pontuação, e repetir

Reuniao 12/04/23

=========== PASSOS DO PROGRAMA:

1. LEITURA DE PARAMETROS EM MODELO:
Parse do arquivo contendo as variáveis a serem testadas
WHILE - Começa a coleta de parametros da eficiencia (CL e CD, coeficiente lift e drag)

2. (WHILE) PARSE:
Refaz o .AVL a partir dos parâmetros

3. (WHILE) OPEREÇÃO DENTRO DO PROGRAMA:
1. LOAD - Carrega o arquivo geometria .AVL
2. OPER 
3. X
4. ST

4. (WHILE) PARSE E PROCESSAMENTO DO ARQUIVO RESULTADO
Recebe o arquivo de saída, faz o parse dos valores calculados
Calcula valor de pontuação (cl, cd ...)
(se não for o primeiro) Calcula valor da derivada entre ponto original e esse no espaço de evaluação
(se estamos no resultado ótimo) FIM
Varia o valor em todas as dimensões por uma porcentagem/valor absoluto coerente (?) 
Volta para o passo 2 com esses dados como o novo arquivo .AVL

5. SAIDA:
basta informar qual foi o arquivo de entrada que otimizou o resultado

========= FIM OPERACAO PROGRAMA

ARQUIVOS:
arquivo geometria -> input
arquivo geometria_otimizado_localmente -> output

DETALHES SOBRE PROCESSO DE EVALUAÇÃO:
parametros do espaço vetorial de pontuação (atualmente, 3)
- pontuacao (saída)
- envergadura (entrada)
- corda (entrada)
isso forma uma superfície, onde temos que encontrar o máximo
analisa um certo ponto e descobre uma "pontuacao"
depois varia 10% cada valor em direções
segue na direção da derivada por uma superfície no pico (variação ao redor do espaço, metodo 100% numerico)
note que o resultado pode ser um máximo local (e tá ok ser, por enquanto)

DETALHES SOBRE CALCULO DO ATRIBUTO PONTUAÇÃO:
- pra uma asa é uma sustentação ideal de cl=2.5 menor arrasto possível
- resultados com CL e CD e vou botar o modulo da diferenca entre o CL e 2 - CD

PERGUNTAS
Como vc consegue o arqvuio? - fazendo na mao, com variaveis

INVESTIGAR
- O que é o comando dentro do AVL chamado RUN CASE?
