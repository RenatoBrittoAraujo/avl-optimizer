# avl-optimizer

Esse programa é um wrapper ao redor do programa [avl](https://www.avl.com/en/simulation-solutions) que automaticamente otimiza parâmetros da estrutura de um veículo que voa no ar. A otimização acontece pelo método numérico de Newton-Raphson, onde uma derivada é calculada sobre uma *função de pontuação* f(x). O valor f'(x), calculado numericamente, pode ser usado para decidir qual dire   

## Arquivos & Pastas

- `avl_optimizer.py`: O próprio, o brabo, o script.
- `geometria.avl`: Arquivo de entrada do AVL, no seu formato original. Edite aqui.
- `otimizado.avl`: Arquivo de saída do script, no seu formato original. Aqui estará o resultado do programa.
- `output.txt`: Saída do programa AVL ao usar comando `ST`.
- `config.json`: Todos as configurações do cálculo.
- `avl_files/`:  Arquivos da internet pra auxiliar o desenvolvimento. Documentação do AVL.
- `target_env/`:  Arquivos que compõem ambiente alvo do avl (aka. uma réplica do seu computador)
- `env/`: Arquivos que compõem ambiente atual que o avl será executado.
- `input_files/`: Arquivos `.avl` de input pra testar o programa.

No arquivo `avl_optmizer.py` é possível encontrar a classe `SumScorer(Scorer)` que contém a função `get_score_from_outfile(x)`. Essa função implementa o sistema de score que será usado para otimizar.

## TODO

- Aileron e flaperon. Template tinha `flaperon` mas esse nome é parametrizado.

## Funcionamento do progama

Note que na pasta `env` contém o executável `avl`, bem como os arquivos `.dat`
e o arquivo de entrada `./t1.avl`.

O script de python instanciando, no mesmo terminal que é rodado, o avl usando
o comando `./avl`. Os comandos são passados por string para o programa via string.

## Funcionamento do AVL

1. Vá pra pasta `./avl_files/runs`
2. Rode `./avl`
3. Digite `LOAD al.avl`
4. Digite `OPER`
5. Digite `X`
6. Digite `ST`
   1. Coloque o path do arquivo de saída, como `test1.txt`

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

### Reuniao 12/04/23

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

### Reuniao 26/04/23

Os inputs são:
| tag                                                                   | range  | significado |
| --------------------------------------------------------------------- | ------ | ----------- |
| `children.surfaces.emp horizontal.children.YDUPLICATE`                | [1, 1] | ??          |
| `children.surfaces.emp horizontal.children.ANGLE`                     | [0, n] | ??          |
| `children.surfaces.emp horizontal.children.TRANSLATE.0`               | [0, n] | X ??        |
| `children.surfaces.emp horizontal.children.TRANSLATE.1`               | [0, n] | Y ??        |
| `children.surfaces.emp horizontal.children.TRANSLATE.2`               | [0, n] | Z ??        |
| `children.surfaces.emp horizontal.children.sections.*.children.Xle`   | [0, n] | ??          |
| `children.surfaces.emp horizontal.children.sections.*.children.Yle`   | [0, n] | ??          |
| `children.surfaces.emp horizontal.children.sections.*.children.Zle`   | [0, n] | ??          |
| `children.surfaces.emp horizontal.children.sections.*.children.Chord` | [0, n] | ??          | **** |

SURFACES A SE ANALISAR:
a idéia é ficar a asa e mexer só na empenagem
pelo menos por agora
a asa vem de aerodin
a range vai ser bem limitada por conta da caixa
a empenagem é a cauda basicamente
saqueei, então ignorar as "wings" só focar nas outras superfícies
emp vertical e horizontal no arquivo
isso, e os flaps também

PORTANTO: **emp horizontal**, **emp vertical** e **flap right**.

Os outputs são:
| tag                  | significado | certeza |
| -------------------- | ----------- | ------- |
| `Cltot`              | ??          | true    |
| `Cl'tot`             | ??          | true    |
| `Cmtot`              | ??          | true    |
| `Cntot`              | ??          | true    |
| `Cn'tot`             | ??          | true    |
| `Neutral point  Xnp` | ??          | true    |
| `CLp`                | ??          | incerto |
| `CLq`                | ??          | incerto |
| `CLr`                | ??          | incerto |
| `CYp`                | ??          | incerto |
| `CYq`                | ??          | incerto |
| `CYr`                | ??          | incerto |
| `Clp`                | ??          | incerto |
| `Clq`                | ??          | incerto |
| `Clr`                | ??          | incerto |
| `Cmp`                | ??          | incerto |
| `Cmq`                | ??          | incerto |
| `Cmr`                | ??          | incerto |
| `Cnp`                | ??          | incerto |
| `Cnq`                | ??          | incerto |
| `Cnr`                | ??          | incerto |

A envergadura consiste em encontrar o ponto mais longe do centro no eixo Y dentro das surfaces. O módulo dessa distância `*2` é igual a envergadura, porque o centro do avião está no centro de Y.

## Notas

Diminuir N-space S-space para aumentar velocidade da simulação

## 27/6/23 - Fórmulas

Lista de limitadores:
\Garantindo estabilidade
Clb Cnr / Clr Cnb  > 1
Cma < 0
Clb < 0
Cnb > 0
\Garantindo controlabiliade, pode ser feito separadamente

|Cm(elevador)| >= 0.03
cld0{X}
|Cl(flaperon)| >= 0.005 !Depende de flaperon mais do que de cauda
|Cn(leme)| >= 0.0012
\Equação de pontuação

P = -0.1*(|Cma - 0.675|) + -0.1(|Cnb - 0.07|) + -0.1(|Clb - 0.07|) + +0.1CLtot + -0.1CDtot + -0.1Cmtot 

tamanho

!Para caso em alpha = 0
