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

#include <omp.h>

using namespace std;
using namespace cv;

//Haarcascades
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String profile_face_cascade_name = "haarcascade_profileface.xml";

//Classificadores
CascadeClassifier face_cascade;
CascadeClassifier profile_face_cascade;

//Janelas
String window_name = "Detecta faces";
String window_name2 = "Reconhecid@";
String window_name3 = "Registrad@";

long int totalfaces = 0, totalprofilefaces = 0, total = 0;
int im_width, im_height, id = 23;
double thresh = 123.0;
vector<Mat> videoProcessado;

Ptr<face::FaceRecognizer> reconhecer = face::createEigenFaceRecognizer(0, thresh);

struct frameEindice
{
	Mat frame;
	int indice;	
};

//------------------------------------------------------------------------------------------------------------------------------------------------------

int reconhece(Mat resized_frame, Rect face){
	Rect rectface = face;
	Mat matface = resized_frame(rectface);
	cvtColor(matface, matface, COLOR_BGR2GRAY);

	Mat resizedmatface;
	resize(matface, resizedmatface, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

	int predicao = reconhecer->predict(resizedmatface);

	if(predicao == id){
		imshow(window_name2, resizedmatface);
		return 1;
	}else{
		return 0;
	}
}

void detectAndDisplay(frameEindice pacote){

	vector<Rect> faces, profile_faces; //Vetor do tipo RECT (contém x, y, width e height)
	Mat resized_frame = pacote.frame;

    //Detecta faces
    face_cascade.detectMultiScale(resized_frame, faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
	profile_face_cascade.detectMultiScale(resized_frame, profile_faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

	totalfaces += faces.size();
	totalprofilefaces += profile_faces.size();
	total = total + faces.size() + profile_faces.size();

	//Para manter elipses coloridas se o frame estiver em escala cinza
	cvtColor(resized_frame, resized_frame, COLOR_GRAY2BGR);

	//rectangle(resized_frame, bodies[i], CV_RGB(0, 0, 255), 2);
	//putText(src, "PROC FRAME", Point(10, 10), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));

	if((faces.size()+profile_faces.size())>7){
		putText(resized_frame, "MULTIDAO", Point(10, 25), CV_FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255));
	}

	//Definir elipses para cada face ou perfil de face detectado
	if(faces.size() > profile_faces.size()){

		for(size_t i = 0; i < faces.size(); i++){
			if(i < profile_faces.size()){ 
				Point center(profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2);
				//Foi reconhecid@?
				if(reconhece(resized_frame, profile_faces[i])){ //Elipse verde
        			ellipse(resized_frame, center, Size(profile_faces[i].width/2, profile_faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
				}else{ //Elipse azul
        			ellipse(resized_frame, center, Size(profile_faces[i].width/2, profile_faces[i].height/2), 0, 0, 360, Scalar(255, 255, 0), 2, 8, 0);
				}
			}
			//Foi reconhecid@?
			Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
			if(reconhece(resized_frame, faces[i])){ //Elipse verde
        		ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        	}else{ //Elipse rosa
	        	ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
    		}
    	}
	} else{
		for(size_t i = 0; i < profile_faces.size(); i++){
			if(i < faces.size()){
				Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
				//Foi reconhecid@?
				if(reconhece(resized_frame, faces[i])){ //Elipse verde
	        		ellipse(resized_frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
	        	}else{ //Elipse rosa
	        		ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
				}
			}
			Point center(profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2);
			//Foi reconhecid@?
			if(reconhece(resized_frame, profile_faces[i])){ //Elipse verde
        		ellipse(resized_frame, center, Size( profile_faces[i].width/2, profile_faces[i].height/2 ), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
			}else{ //Elipse azul
	        	ellipse(resized_frame, center, Size( profile_faces[i].width/2, profile_faces[i].height/2 ), 0, 0, 360, Scalar(255, 255, 0), 2, 8, 0);
    		}
    	}		
	}

	while((videoProcessado.size() < pacote.indice)){}
	videoProcessado.push_back(resized_frame);
}

//------------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv){

	VideoCapture capture("video.mp4");
	Mat frame, img, face;
	vector<Rect> rect;
	double fps;
	int totalframes;
	vector<Mat> imagens; //Vetor de imagens para treinar o algoritmo reconhecedor de faces
	vector<int> tags; //Vetor de tags para o reconhecedor de faces

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

	//Confere se recebeu strings com nomes das imagens
	if( argc < 2 ){
		cout << "Nao ha imagens suficientes, argc = " << argc << endl;
		return -1;
	}

	/*Lê os nomes das imagens (de uma mesma pessoa) digitados no terminal (dps do ./exe)
	Converte cor, encontra face, redimensiona e insere nos vetores imagens
	Insere id's no vetor tags*/
	for(int k=1; k<argc; k++){

		img = imread(argv[k], IMREAD_GRAYSCALE);

		if(img.empty()){
			cout << "Impossivel ler imagem" << endl;
			return -1;
		}

    	//equalizeHist(img, img);

		face_cascade.detectMultiScale(img, rect, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

		face = img(rect[0]);

		if(k==1){
			im_width = face.cols;
			im_height = face.rows;
		}

		resize(face, face, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC); //Padroniza tamanho das matrizes

		imagens.push_back(face); //Registra faces no vetor de matrizes
		tags.push_back(id);
		rect.clear();
		imshow(window_name3, face);
		char p = waitKey(1000);
	}

	 //Printa última face registrada

	//Treina o reconhecedor de faces com os vetores imagens e tags
    reconhecer->train(imagens, tags);


    //Mat videoProcessado[totalframes];
	    //Lê vídeo frame por frame, detecta faces e reconhece
    	int i = 0;
    	int j = 0;
    	int lendo = 1;
    	int printando = 1;

    	while(lendo || printando){

    		if(i < totalframes){
	    		capture.read(frame);
	    		i++;

	    		if(frame.empty()){
					printf( "Impossivel ler frame\n" );
					break;
				}

		    	//Redimensiona frame para reduzir o delay na reproducao do video durante a deteccao
				const float scale = 2.5;
				Mat resized_frame(cvRound(frame.rows / scale), cvRound(frame.cols / scale), CV_8UC1);
				resize( frame, resized_frame, resized_frame.size() );

				//Converte para escala cinza, ajusta brilho e equaliza frame
	    		cvtColor(resized_frame, resized_frame, COLOR_BGR2GRAY);
	    		//equalizeHist(resized_frame, resized_frame);

	    		frameEindice pacote;
	    		pacote.frame = resized_frame;
	    		pacote.indice = (i-1); //Declara struct com indice e frame a ser processado

				thread (detectAndDisplay, pacote).detach();
			}else{
				lendo = 0;
			}

			Mat show;
			if(videoProcessado.size() > j && j < totalframes){
				show = videoProcessado[j];
				imshow(window_name, show);
				j++;
			}else if(j >= totalframes){
				printando = 0;
			}
			
			char c = waitKey(1);
			if(c == 32){
				break;
			} //Interrompe deteccao de faces e reproducao do video ao clicar espaco
		}

	destroyAllWindows();
	capture.release(); 

	cout << "faces: " << totalfaces << endl; //Total de faces (frontais)
	cout << "profile faces: " << totalprofilefaces << endl; //Total de perfis de faces
	cout << "total: " << total << endl;

	return 0;
}