#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <ctime>

using namespace std;
using namespace cv;
using namespace std::chrono;

//Haarcascades
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String profile_face_cascade_name = "haarcascade_profileface.xml";

//Classificadores
CascadeClassifier face_cascade;
CascadeClassifier profile_face_cascade;

//Janelas
String window_name = "Detecta faces, reconhece faces e identifica multidao";

//Variaveis globais
int im_width, im_height, minMultidao = 10;
float areaResizedFrame = 0, razaoFacesFrame = 0.2, scale = 2.5;
double thresh = 123.0;
Size size(40,40);
vector<Mat> videoProcessado;
float totalPorFaixa[8] = {}, totalContabilizado = 0;

Ptr<face::FaceRecognizer> reconhecer = face::createFisherFaceRecognizer(0, thresh);

std::mutex mutex1, mutex2;

struct frameEindice
{
	Mat frame;
	int indice;
};

//FUNCOES
//------------------------------------------------------------------------------------------------------------------------------------------------------
void estatisticaTomDaPele(){
	float p;
	FILE * dados;

	dados = fopen("dados.txt", "wt");

	if(dados == NULL){
		cout << "Impossivel acessar arquivo\n";
	}else{
	//Calcula porcentagens por faixa de intensidade de cor da pele
		for(int i = 0; i < 8; i++){
			p = totalPorFaixa[i]/totalContabilizado;
			if(p > 0.0009){
				p = p*100;
				fprintf(dados, "%.1f por cento das faces detectadas possuem intensidade da cor da pele entre %d e %d\n", p, i*32, i*32+32);
			}
		}

		fclose (dados);
	}
}

void CorPrincipal(Mat face){
	Mat data;
	face.convertTo(data, CV_32F);
	Vec3f cor;
	float pixels = 0;

	float azul = 0, verde = 0, vermelho = 0;

	long int i, j;
	for(i = 0; i < data.cols; i ++){
		for(j = 0; j < data.rows; j++){
			cor = data.at<Vec3f>(j, i);
			azul += cor.val[0]*cor.val[0];
			verde += cor.val[1]*cor.val[1];
			vermelho += cor.val[2]*cor.val[2];
			pixels++;
		}
	}

	azul = sqrt(azul/pixels);
	verde = sqrt(verde/pixels);
	vermelho = sqrt(vermelho/pixels);

	data.convertTo(data, CV_8UC3); //3 canais (BGR)
	Rect retang(0,0,10,10);
	if(data.rows > 10 && data.cols > 10){
		rectangle(data, retang, Scalar(azul, verde, vermelho), -1);
		Mat exibecor (data, Rect(0, 0, 10, 10));

		cvtColor(exibecor, exibecor, COLOR_BGR2GRAY); //1 canal

		Scalar intensidade = exibecor.at<uchar>(0, 0); //Le intensidade de um pixel da matriz monocromatica
		int i = intensidade.val[0];
		int faixa = 32;
		int destino;

		destino = i/faixa;

		mutex2.lock();
		totalPorFaixa[destino]++;
		totalContabilizado++;
		mutex2.unlock();
	}
}

int reconhece(Mat resized_frame, Rect face){
	Rect rectface = face;
	Mat matface = resized_frame(rectface);
	cvtColor(matface, matface, COLOR_BGR2GRAY);

	Mat resizedmatface;
	resize(matface, resizedmatface, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

	int predicao = reconhecer->predict(resizedmatface);

	if(predicao >= 0){ //Esta nos registros
		return predicao;
	}else{
		return 0;
	}
}

void detectAndDisplay(frameEindice pacote){

	vector<Rect> faces, profile_faces; //Vetor do tipo RECT (contém x, y, width e height)
	Mat resized_frame = pacote.frame;

    //Detecta faces
    face_cascade.detectMultiScale(resized_frame, faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, size);
	profile_face_cascade.detectMultiScale(resized_frame, profile_faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, size);

	float areaFaces = 0;
	int idReconhecidoInt = -1;
	std::string idReconhecidoStr = "";

	//Definir elipses para cada face ou perfil de face detectado e/ou reconhecido
	if(faces.size() > profile_faces.size()){

		for(size_t i = 0; i < faces.size(); i++){
			if(i < profile_faces.size()){ 
				Point center(profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2);
				//Foi reconhecid@?
				idReconhecidoInt = reconhece(resized_frame, profile_faces[i]);
				if(idReconhecidoInt > 0){ //Elipse verde
					idReconhecidoStr = std::to_string(idReconhecidoInt);
					putText(resized_frame, idReconhecidoStr, Point(profile_faces[i].x - profile_faces[i].height/2, profile_faces[i].y), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
        			ellipse(resized_frame, center, Size(profile_faces[i].width/2, profile_faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
				}else{ //Elipse azul
        			ellipse(resized_frame, center, Size(profile_faces[i].width/2, profile_faces[i].height/2), 0, 0, 360, Scalar(255, 255, 0), 2, 8, 0);
				}

				areaFaces += profile_faces[i].width * profile_faces[i].height;
			}
			CorPrincipal(resized_frame(faces[i]));
			Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
			//Foi reconhecid@?
			idReconhecidoInt = reconhece(resized_frame, faces[i]);
			if(idReconhecidoInt > 0){ //Elipse verde
				idReconhecidoStr = std::to_string(idReconhecidoInt);
				putText(resized_frame, idReconhecidoStr, Point(faces[i].x - faces[i].height/2, faces[i].y), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
        		ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        	}else{ //Elipse rosa
	        	ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
    		}
    		areaFaces += faces[i].width * faces[i].height;
    	}
	} else{
		for(size_t i = 0; i < profile_faces.size(); i++){
			if(i < faces.size()){
				CorPrincipal(resized_frame(faces[i]));
				Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
				//Foi reconhecid@?
				idReconhecidoInt = reconhece(resized_frame, faces[i]);
				if(idReconhecidoInt > 0){ //Elipse verde
					idReconhecidoStr = std::to_string(idReconhecidoInt);
					putText(resized_frame, idReconhecidoStr, Point(faces[i].x - faces[i].height/2, faces[i].y), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
        			ellipse(resized_frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
	        	}else{ //Elipse rosa
	        		ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
				}
				areaFaces += faces[i].width * faces[i].height;
			}
			Point center(profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2);
			//Foi reconhecid@?
			idReconhecidoInt = reconhece(resized_frame, profile_faces[i]);
			if(idReconhecidoInt > 0){ //Elipse verde
				idReconhecidoStr = std::to_string(idReconhecidoInt);
				putText(resized_frame, idReconhecidoStr, Point(profile_faces[i].x - profile_faces[i].height/2, profile_faces[i].y), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
        		ellipse(resized_frame, center, Size( profile_faces[i].width/2, profile_faces[i].height/2 ), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
			}else{ //Elipse azul
	        	ellipse(resized_frame, center, Size( profile_faces[i].width/2, profile_faces[i].height/2 ), 0, 0, 360, Scalar(255, 255, 0), 2, 8, 0);
    		}
    		areaFaces += profile_faces[i].width * profile_faces[i].height;
    	}
	}

	if((faces.size() + profile_faces.size()) > minMultidao && (areaFaces / areaResizedFrame) >= razaoFacesFrame){
		putText(resized_frame, "MULTIDAO", Point(10, 25), CV_FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255));
	}

	mutex1.lock();
	while((videoProcessado.size() < pacote.indice)){
		mutex1.unlock();
		mutex1.lock();
	}
	videoProcessado.push_back(resized_frame);
	mutex1.unlock();
}

//MAIN
//------------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv){

	VideoCapture capture("video.mp4");
	double fps;
	int totalframes;

    if(!face_cascade.load(face_cascade_name)){
    	printf("Erro ao carregar face cascade\n");
    	return -1;
    }

    if(!profile_face_cascade.load(profile_face_cascade_name)){
    	printf("Erro ao carregar profile face cascade\n");
    	return -1;
    }

    if(!capture.isOpened()) {
    	printf("Erro ao abrir o video\n"); 
    	return -1;
    }

    fps = capture.get(CV_CAP_PROP_FPS);

    totalframes = (capture.get(CAP_PROP_FRAME_COUNT));

	/*Lê os nomes das imagens e suas tags em arquivo txt
	Converte cor, encontra face, redimensiona e insere no vetor imagens
	Insere tag no vetor tags*/
	FILE * imgtag;
	imgtag = fopen("imgtag.txt", "rt");

	if(imgtag == NULL){
		cout << "Impossivel ler arquivo\n";
		return -1;
	}

	char nomeimg[20];
	int tag;
	int k = 0;
	Mat img, face;
	vector<Rect> rect;
	vector<Mat> imagens; //Vetor de imagens para treinar o algoritmo reconhecedor de faces
	vector<int> tags; //Vetor de tags para o reconhecedor de faces

	while(fscanf(imgtag, "%s%d", nomeimg, &tag) != EOF){

		img = imread(nomeimg, IMREAD_GRAYSCALE);

		if(img.empty()){
			cout << "Impossivel ler imagem" << endl;
			return -1;
		}

		face_cascade.detectMultiScale(img, rect, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

		face = img(rect[0]);

		if(k==0){
			im_width = face.cols;
			im_height = face.rows;
		}

		resize(face, face, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC); //Padroniza tamanho das matrizes

		imagens.push_back(face); //Registra faces no vetor de matrizes
		tags.push_back(tag);
		rect.clear();
		k++;
	}
	fclose (imgtag);

	//Treina o reconhecedor de faces com os vetores imagens e tags
    reconhecer->train(imagens, tags);

    //AVISOS AO USUARIO
    cout << "---------------- INSTRUCOES ----------------" << endl;
    cout << "Pressione espaco caso queira pausar/dar play no video;" << endl;
    cout << "Presione 'i' caso queira interromper o processamento e reproducao do video;" << endl;
    cout << "Ao fim da execucao, verifique as estatisticas sobre tom da pele das faces detectadas no arquivo 'dados,txt'." << endl;

	std::chrono::time_point<std::chrono::system_clock> comeco, fim;
	double intervalo;
	int dif;
	int i = 0;
	int j = 0;
	int lendo = 1;
	int printando = 1;
	int pausa = 0; //Clicar espaco para pausar exibicao
	Mat frame, show;
    char c;
    float tempoEntreFrames = 1000 / fps;
    //Lê vídeo frame por frame, detecta faces, reconhece face e projeta frames ja processados na janela
   	while(lendo || printando){
   		if(i < totalframes){
    		capture.read(frame);
    		i++;

    		if(frame.empty()){
				printf( "Impossivel ler frame\n" );
				break;
			}

	    	//Redimensiona frame para reduzir o delay na reproducao do video durante a deteccao
			Mat resized_frame(cvRound(frame.rows / scale), cvRound(frame.cols / scale), CV_8UC1);
			resize( frame, resized_frame, resized_frame.size() );
			areaResizedFrame = resized_frame.rows * resized_frame.cols; //Area do frame a ser processado

    		frameEindice pacote; //Struct contem Mat e Int
    		pacote.frame = resized_frame;
    		pacote.indice = (i-1); //Declara struct com indice e frame a ser processado

			thread (detectAndDisplay, pacote).detach(); //nova linha de execucao para processar frame
		}else{
			lendo = 0;
		}

		//Ha frames processados para exibicao e video nao esta pausado?
		if(videoProcessado.size() > j && j < totalframes && pausa == 0){
			show = videoProcessado[j];
			if(j == 0){
				imshow(window_name, show);
				comeco = high_resolution_clock::now(); //Le tempo para comparar com o do proximo frame
			}else{
				fim = high_resolution_clock::now(); //Le tempo para comparar com o do frame anterior

				//Intervalo de tempo entre exibicao de dois frames consecutivos
				intervalo = std::chrono::duration_cast<std::chrono::milliseconds>(fim-comeco).count();

				//Eh menor que o tempo necessario para manter o FPS original do video?
				if(intervalo < tempoEntreFrames){
					dif = tempoEntreFrames - intervalo;
					std::this_thread::sleep_for(std::chrono::milliseconds(dif)); //Espera para atingir FPS correto
					imshow(window_name, show);
				}else{
					imshow(window_name, show);
				}
				comeco = high_resolution_clock::now();
			}
			j++;
		//Fim dos frames processados?
		}else if(j >= totalframes){
			printando = 0;
		}

		c = waitKey(1);
		if(c == 32){ //Da play ou pausa ao clicar espaco
			if(pausa){
				pausa = 0;
			}else if(pausa == 0){
				pausa = 1;
			}
		}else if(c == 105){ //Interrompe processamento de frames e exibicao do video ao clicar i
			lendo = 0;
			printando = 0;
		}
	}

	destroyAllWindows();
	capture.release();

	estatisticaTomDaPele();

	return 0;
}