
## Objetivo

O objetivo deste produto é fornecer um serviço automatizado que recomenda leads para um usuário dado sua atual lista de clientes (Portfólio).

## Contextualização

Algumas empresas gostariam de saber quem são as demais empresas em um determinado mercado (população) que tem maior probabilidade se tornarem seus próximos clientes. Ou seja, a sua solução deve encontrar no mercado quem são os leads mais aderentes dado as características dos clientes presentes no portfólio do usuário.

Além disso, sua solução deve ser agnóstica ao usuário. Qualquer usuário com uma lista de clientes que queira explorar esse mercado pode extrair valor do serviço.

Para o desafio, deverão ser consideradas as seguintes bases:

Mercado: Base com informações sobre as empresas do Mercado a ser considerado. Portfolio 1: Ids dos clientes da empresa 1 Portfolio 2: Ids dos clientes da empresa 2 Portfolio 3: Ids dos clientes da empresa 3

Obs: todas as empresas(ids) dos portfolios estão contidos no Mercado(base de população).

Link para download das bases Mercado, Portfolio 1, Portfolio 2 e Portfolio 3 respectivamente:

[https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_market.csv.zip](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_market.csv.zip)

[https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio1.csv](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio1.csv)

[https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio2.csv](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio2.csv)

[https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio3.csv](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio3.csv)

As bases de portfólio poderão ser utilizadas para testar a aderência da solução. Além disso, se a equipe desejar, poderá simular portfólios por meio de amostragens no mercado.

[Descrição de variáveis](https://s3-us-west-1.amazonaws.com/codenation-challenges/ml-leads/features_dictionary.pdf)

## Requisitos técnicos obrigatórios

-   Utilizar técnicas de data science e machine learning para desenvolver o projeto;
-   Apresentar o desenvolvimento e outputs do modelo em um Jupyter Notebook ou outra tecnologia de apresentação de Output de modelos de Machine Learning;
-   A análise deve considerar os seguintes pontos: análise exploratória dos dados, tratamento dos dados, avaliação de algoritmos, treinamento do modelo, avaliação de performance do modelo e visualização dos resultados;
-   Para a apresentação do projeto, o tempo entre o treinamento do modelo e o output deve ser menor que 20 min.

### Sobre a apresentação do projeto

Para que todos do grupo tenham a chance de apresentar seu trabalho, a  **apresentação**  deve ser feita de forma  **individual**. Vocês podem ensaiar juntos, fazer um roteiro parecido, mas é  **importante que cada participante faça sua própria gravação**.

Objetividade é muito importante - falem naturalmente e sem ler, por favor! :) Recomendamos que você faça um video-call e gravem este call. Assim, poderão ficar à vontade para compartilhar a tela e mostrar o código ou qualquer outra coisa importante. O vídeo deve ter no máximo  **10 minutos**.

Segue uma  **sugestão**  de roteiro:

#### 1- Apresentação pessoal

-   “Oi, pessoal…, eu me chamo _____ e vou apresentar para vocês o projeto final que fiz com a squad ______(número e nome) da Aceleração _______ da Codenation.”

#### 2- Apresentação do projeto

-   Os membros da squad;
-   Descrição do projeto e desenvolvimento do processo que a squad utilizou para resolver o problema;
-   Divisão de tarefas entre os membros da squad e quais foram suas principais responsabilidades;
-   Tecnologias utilizadas e eficácia;
-   Aprendizados destacados durante o processo e ao final do projeto;
-   Duas principais dificuldades, e como foram contornadas;
-   Dois principais acertos ou escolhas acertadas que fizeram diferença no projeto e por quê.

Para ficar mais fácil,  [dê uma olhada nesta apresentação de projeto de um programa que realizamos em Joinville](https://mkt.codenation.com.br/r/f00c7f5ea676691d7cf64ad61?ct=YTo1OntzOjY6InNvdXJjZSI7YToyOntpOjA7czo1OiJlbWFpbCI7aToxO2k6MjE4O31zOjU6ImVtYWlsIjtpOjIxODtzOjQ6InN0YXQiO3M6MTM6IjVjOWJhOGRmZWY1NTYiO3M6NDoibGVhZCI7czo1OiIyMjQ1OSI7czo3OiJjaGFubmVsIjthOjE6e3M6NToiZW1haWwiO2k6MjE4O319&utm_source=softplan&utm_medium=email&utm_campaign=Lista-participantes-POC). Neste caso, os participantes desenvolveram em squads uma aplicação (backend e frontend) que buscava anunciar animais perdidos ou animais para adoção. Assim, pessoas interessadas poderiam colaborar para adotar e/ou encontrar um pet. Fiz alguns comentários na apresentação para ajudar vocês! :)

#### Como enviar os vídeos?

Após terem determinado o roteiro e feito suas gravações individuais, encaminhe por e-mail o link do  **vídeo no YouTube**  (lembre de colocá-lo como  **não listado**, por favor). No título, use a seguinte descrição:  **“Apresentação projeto final [Seu Nome] [nome de sua squad]”**. O link do vídeo deve ser enviado para  [mario.machado@codenation.dev](mailto:mario.machado@codenation.dev)  e  [ingrid.adam@codenation.dev](mailto:ingrid.adam@codenation.dev), juntamente com o link do Code Review do projeto na plataforma Codenation, com o assunto  **“AceleraDev Python - Squad [nº da sua squad]”**  até o dia  **14/08/2019**.