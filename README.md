# Detecção, Reconhecimento, Rastreamento e Leitura de tom de pele facial + Identificação de Multidão


Projeto desenvolvido em C++ com uso da biblioteca OpenCV (versão 3.2.0)

# Compilação e Execução
Para compilar o projeto, instale a versão 3.2.0 da biblioteca OpenCV, mantenha todos os arquivos deste repositório no mesmo diretório e digite no terminal (do diretório onde está localizado o projeto):

    $ cmake .
    $ make

Para executar, digite:

    $ ./Deteccao

# O que acontece durante a execução?
  - Leitura de nomes de imagens e do ID de cada imagem no arquivo *imgtag.txt*
  - As faces presentes nas imagens e seus ID's são "registrados" e usados para treinar um algoritmo de reconhecimento facial
  - O arquivo *video.mp4* é lido frame por frame
  - Cada frame passa por um processo de: Identificação de multidão, Detecção facial, Reconhecimento facial e Leitura aproximada da intensidade da cor da pele de cada face detectada
  - Os frames processados são exibidos em uma janela (aproximadamente com o mesmo FPS do *video.mp4*) identificando com elipses azuis os perfis de faces detectados, com elipses rosas as faces frontais detectadas, com elipses verdes faces reconhecidas (com seus ID's correspondentes) e com um alerta textual a presença de uma multidão
  - Ao fim do processamento e da exibição dos frames, as informações obtidas sobre os tons de pele identificados são escritas no arquivo *dados.txt*
   
   
