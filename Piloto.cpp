#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

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
String window_name = "Detecta faces, reconhece face e identifica multidao";
String window_name2 = "Reconhecid@";
String window_name3 = "Registrad@";

//Variaveis globais
long int totalfaces = 0, totalprofilefaces = 0, total = 0;
int im_width, im_height, id = 23, minMultidao = 10;
float areaResizedFrame = 0, razaoFacesFrame = 0.2, scale = 2.5;
double thresh = 123.0;
Size size(40,40);
vector<Mat> videoProcessado;

Ptr<face::FaceRecognizer> reconhecer = face::createFisherFaceRecognizer(0, thresh);

std::mutex mutex1, mutex2, mutex3, mutex4;

struct frameEindice
{
	Mat frame;
	int indice;
};

//FUNCOES
//------------------------------------------------------------------------------------------------------------------------------------------------------
void CorPrincipal(Mat face){
	Mat data;
	face.convertTo(data, CV_32F);
	Vec3f intensidade;
	float pixels = 0;

	float azul = 0, verde = 0, vermelho = 0;

	long int i, j;
	for(i = 0; i < data.cols; i ++){
		for(j = 0; j < data.rows; j++){
			intensidade = data.at<Vec3f>(j, i);
			if(intensidade.val[0] > 20 && intensidade.val[1] > 40 && intensidade.val[2] > 80){ //tonalidade da cor da pele
				azul += intensidade.val[0];
				verde += intensidade.val[1];
				vermelho += intensidade.val[2];
				pixels++;
			}
		}
	}

	azul = azul/pixels;
	verde = verde/pixels;
	vermelho = vermelho/pixels;
	//intensidade.val[0] = azul;
	//intensidade.val[1] = verde;
	//intensidade.val[2] = vermelho;

	//cout << intensidade << endl;

	data.convertTo(data, CV_8UC3);
	Rect retang(0,0,80,80);
	if(data.rows > 80 && data.cols > 80){
		rectangle(data, retang, Scalar(azul, verde, vermelho), -1);
		Mat exibecor (data, Rect(0, 0, 80, 80));
		mutex4.lock();
		imshow("Cor", exibecor);
		mutex4.unlock();
	}
}

int reconhece(Mat resized_frame, Rect face){
	Rect rectface = face;
	Mat matface = resized_frame(rectface);
	cvtColor(matface, matface, COLOR_BGR2GRAY);

	Mat resizedmatface;
	resize(matface, resizedmatface, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

	int predicao = reconhecer->predict(resizedmatface);

	if(predicao == id){
		mutex2.lock();
		imshow(window_name2, resizedmatface);
		mutex2.unlock();
		return 1;
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

	mutex1.lock();
	totalfaces += faces.size();
	totalprofilefaces += profile_faces.size();
	total = total + faces.size() + profile_faces.size();
	mutex1.unlock();

	float areaFaces = 0;
	//Definir elipses para cada face ou perfil de face detectado
	if(faces.size() > profile_faces.size()){

		for(size_t i = 0; i < faces.size(); i++){
			if(i < profile_faces.size()){ 
				Point center(profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2);
				//Foi reconhecid@?
				if(reconhece(resized_frame, profile_faces[i])){ //Elipse verde
					CorPrincipal(resized_frame(profile_faces[i]));
					//putText(resized_frame, "Identificad@", Point(profile_faces[i].x - profile_faces[i].height/2, profile_faces[i].y), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
        			ellipse(resized_frame, center, Size(profile_faces[i].width/2, profile_faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
				}else{ //Elipse azul
        			ellipse(resized_frame, center, Size(profile_faces[i].width/2, profile_faces[i].height/2), 0, 0, 360, Scalar(255, 255, 0), 2, 8, 0);
				}

				areaFaces += profile_faces[i].width * profile_faces[i].height;
			}

			Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
			//Foi reconhecid@?
			if(reconhece(resized_frame, faces[i])){ //Elipse verde
				CorPrincipal(resized_frame(faces[i]));
				ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        	}else{ //Elipse rosa
	        	ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
    		}
    		areaFaces += faces[i].width * faces[i].height;
    	}
	} else{
		for(size_t i = 0; i < profile_faces.size(); i++){
			if(i < faces.size()){
				Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
				//Foi reconhecid@?
				if(reconhece(resized_frame, faces[i])){ //Elipse verde
					CorPrincipal(resized_frame(faces[i]));
					ellipse(resized_frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
	        	}else{ //Elipse rosa
	        		ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
				}
				areaFaces += faces[i].width * faces[i].height;
			}
			Point center(profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2);
			//Foi reconhecid@?
			if(reconhece(resized_frame, profile_faces[i])){ //Elipse verde
				CorPrincipal(resized_frame(profile_faces[i]));
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

	mutex3.lock();
	while((videoProcessado.size() < pacote.indice)){
		mutex3.unlock();
		mutex3.lock();
	}
	videoProcessado.push_back(resized_frame);
	mutex3.unlock();
}

//MAIN
//------------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv){

	VideoCapture capture("video.mp4");
	double fps;
	int totalframes;

    if(!face_cascade.load(face_cascade_name)){
    	printf("Error ao carregar face cascade\n");
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
    cout << "fps do video: " << fps << endl;


    totalframes = (capture.get(CAP_PROP_FRAME_COUNT));
	cout << "total de frames no video: " << totalframes << endl;

	//Confere se recebeu string com nome do arquivo txt (contendo imagens e tags)
	if(argc < 2){
		cout << "Nao ha parametros suficientes, argc = " << argc << endl;
		return -1;
	}

	/*Lê os nomes das imagens e suas tags em arquivo txt
	Converte cor, encontra face, redimensiona e insere no vetor imagens
	Insere tag no vetor tags*/
	FILE * imgtag;
	imgtag = fopen(argv[1], "rt");

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
    	//equalizeHist(img, img);

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

		if(tag == id){
			imshow(window_name3, face); //Printa a face registrada correspondente ao ID procurado
		}
		k++;
	}
	fclose (imgtag);

	//Treina o reconhecedor de faces com os vetores imagens e tags
    reconhecer->train(imagens, tags);


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

	cout << "faces: " << totalfaces << endl; //Total de faces (frontais)
	cout << "profile faces: " << totalprofilefaces << endl; //Total de perfis de faces
	cout << "total: " << total << endl;

	return 0;
}