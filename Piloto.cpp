#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace cv;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String profile_face_cascade_name = "haarcascade_profileface.xml";
CascadeClassifier face_cascade;
CascadeClassifier profile_face_cascade;
String window_name = "Detecta faces";

long int totalfaces = 0, totalprofilefaces = 0;

void detectAndDisplay(Mat frame){
	vector<Rect> faces, profile_faces; //Vetor do tipo RECT (contém x, y, width e height)

	//Redimensiona frame para reduzir o delay na reproducao do video durante a deteccao
	const float scale = 2;
	cv::Mat resized_frame( cvRound( frame.rows / scale ), cvRound( frame.cols / scale ), CV_8UC1 );
	cv::resize( frame, resized_frame, resized_frame.size() );

    //Detecta faces
    face_cascade.detectMultiScale( resized_frame, faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
	profile_face_cascade.detectMultiScale( resized_frame, profile_faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

	totalfaces += faces.size();
	totalprofilefaces += profile_faces.size();


	if( faces.size() > profile_faces.size() ){

		for( size_t i = 0; i < faces.size(); i++ ){
			if( i < profile_faces.size() ){
				Point center( profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2 );
        		ellipse( resized_frame, center, Size( profile_faces[i].width/2, profile_faces[i].height/2 ), 0, 0, 360, Scalar( 255, 255, 0 ), 4, 8, 0 );
			}
        	Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        	ellipse( resized_frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    	}

	} else{

		for( size_t i = 0; i < profile_faces.size(); i++ ){

			if( i < faces.size() ){
				Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        		ellipse( resized_frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
			}

       		Point center( profile_faces[i].x + profile_faces[i].width/2, profile_faces[i].y + profile_faces[i].height/2 );
        	ellipse( resized_frame, center, Size( profile_faces[i].width/2, profile_faces[i].height/2 ), 0, 0, 360, Scalar( 255, 255, 0 ), 4, 8, 0 );
    	}		

	}

    imshow( window_name, resized_frame );
}



int main (int argc, char** argv){
	VideoCapture capture("video.mp4");
	Mat frame;

    if( !face_cascade.load( face_cascade_name ) ){
    	printf("Error ao carregar face cascade\n");
    	return -1;
    }

    if( !profile_face_cascade.load( profile_face_cascade_name)){
    	printf("Erro ao carregar profile face cascade\n");
    	return -1;
    }

    if( !capture.isOpened() ) {
    	printf("Erro ao abrir o video\n"); 
    	return -1;
    }
    
    while( capture.read(frame) ){

        if( frame.empty() ){
            printf("Impossivel ler frame\n");
            break;
        }

        detectAndDisplay(frame);
        char c = (char)waitKey(10);

        if( c == 32 ){
       		break;
       	} //Interrompe deteccao de faces e reproducao do video ao clicar espaco
	
	}

	cout << "faces: " << totalfaces << endl; //Total de faces (frontais)
	cout << "profile faces: " << totalprofilefaces << endl; //Total de perfis de faces

	return 0;
}