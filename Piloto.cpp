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
CascadeClassifier face_cascade;
String window_name = "Detecta faces";

void detectAndDisplay(Mat frame){
	vector<Rect> faces; // vetor do tipo RECT (contém x, y, width e height)

    //Detecta faces
    face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    }

    imshow( window_name, frame );
}



int main (int argc, char** argv){
	VideoCapture capture("video.mp4");
	Mat frame;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };

    //capture.open( -1 ); Para detectar da webcam
    if ( !capture.isOpened() ) {
    	printf("Erro ao abrir o video\n"); 
    	return -1;
    }
    
    while(capture.read(frame)){

        if(frame.empty()){
            printf("Impossivel ler frame\n");
            break;
        }

        detectAndDisplay(frame);
        char c = (char)waitKey(10);

        if( c == 32 ){
       		break;
       	} // interrompe reproducao do video ao clicar espaco
	
	}
	return 0;
}