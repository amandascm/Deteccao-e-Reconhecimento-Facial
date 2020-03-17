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

#include <omp.h>

using namespace std;
using namespace cv;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String profile_face_cascade_name = "haarcascade_profileface.xml";
CascadeClassifier face_cascade;
CascadeClassifier profile_face_cascade;
String window_name = "Detecta faces";
String window_name2 = "Reconhecido";
String window_name3 = "Registrado";

long int totalfaces = 0, totalprofilefaces = 0, total = 0;
int im_width, im_height, num_componentes = 1;
double thresh = 10.0;

Ptr<face::FaceRecognizer> reconhecer = face::createEigenFaceRecognizer(num_componentes, thresh);

//----------------------------------------------------------------------------------------------------------------------


void detectAndDisplay(Mat frame){

	vector<Rect> faces, profile_faces; //Vetor do tipo RECT (contém x, y, width e height)

	//Redimensiona frame para reduzir o delay na reproducao do video durante a deteccao
	const float scale = 2;
	Mat resized_frame(cvRound(frame.rows / scale), cvRound(frame.cols / scale), CV_8UC1);
	resize( frame, resized_frame, resized_frame.size() );

	//Converte para escala cinza, ajusta brilho e equaliza frame
	Mat frame_gray;
    cvtColor(resized_frame, resized_frame, COLOR_BGR2GRAY);
    equalizeHist(resized_frame, resized_frame);

    //Detecta faces
    face_cascade.detectMultiScale(resized_frame, faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
	profile_face_cascade.detectMultiScale(resized_frame, profile_faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

	//Contabiliza total de faces frontais e de perfis de faces
	totalfaces += faces.size();
	totalprofilefaces += profile_faces.size();
	total = total + faces.size() + profile_faces.size();

	//Para manter elipses coloridas se o frame estiver em escala cinza
	cvtColor(resized_frame, resized_frame, COLOR_GRAY2BGR);

	//Definir elipses para cada face ou perfil de face detectado
	if(faces.size() > profile_faces.size()){

		for(size_t i = 0; i < faces.size(); i++){
			if(i < profile_faces.size()){ //Elipse azul
				Point center(profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2);
        		ellipse(resized_frame, center, Size(profile_faces[i].width/2, profile_faces[i].height/2), 0, 0, 360, Scalar(255, 255, 0), 2, 8, 0);
			}

			Rect rectface = faces[i];
			Mat matface = resized_frame(rectface);
			cvtColor( matface, matface, COLOR_BGR2GRAY );
			Mat resizedmatface;

			resize(matface, resizedmatface, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			int predicao = reconhecer->predict(resizedmatface);
			//Foi reconhecido?
			if(predicao == 23){ //Elipse verde
				Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
        		ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        		imshow(window_name2, resizedmatface);
        	}else{ //Elipse rosa
	        	Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
	        	ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
    		}
    	}
	} else{

		for(size_t i = 0; i < profile_faces.size(); i++){
			if(i < faces.size()){
				Rect rectface = faces[i];
				Mat matface = resized_frame(rectface);
				cvtColor( matface, matface, COLOR_BGR2GRAY );
				Mat resizedmatface;

				resize(matface, resizedmatface, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
				int predicao = reconhecer->predict(resizedmatface);
				//Foi reconhecido?
				if(predicao == 23){ //Elipse verde
					Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
	        		ellipse(resized_frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
	        		imshow(window_name2, resizedmatface);
	        	}else{ //Elipse rosa
					Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
	        		ellipse(resized_frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
				}
			}
			//Elipse azul
       		Point center(profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2);
        	ellipse(resized_frame, center, Size( profile_faces[i].width/2, profile_faces[i].height/2 ), 0, 0, 360, Scalar(255, 255, 0), 2, 8, 0);
    	}		
	}

	//Printa o frame na janela
    imshow(window_name, resized_frame);
}

//----------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv){

	VideoCapture capture("video.mp4");
	Mat frame, img;
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
	for(int i=1; i<argc; i++){
		img = imread(argv[i], IMREAD_GRAYSCALE);

		if(img.empty()){
			cout << "Impossivel ler imagem" << endl;
			return -1;
		}

    	equalizeHist(img, img);

		face_cascade.detectMultiScale(img, rect, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

		Mat face = img(rect[i-1]);

		if(i==1){
			im_width = face.cols;
			im_height = face.rows;
		}

		resize(face, face, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

		imshow(window_name3, face);

		imagens.push_back(face);
		//tags.push_back((i-1));
		tags.push_back(23);
	}

	//treina o reconhecedor de faces com os vetores imagens e tags
    reconhecer->train(imagens, tags);

    	//Lê vídeo frame por frame, detecta faces e reconhece
    	while(capture.read(frame)){
    		
		    if(frame.empty()){
		        printf( "Impossivel ler frame\n" );
		        break;
		    }

		    detectAndDisplay(frame);

		    //imshow( window_name, frame );
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