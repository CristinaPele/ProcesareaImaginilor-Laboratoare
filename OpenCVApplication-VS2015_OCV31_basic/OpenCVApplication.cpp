#include "stdafx.h"
#include "common.h"
#include <queue>
#include <iostream>


bool equalColor(Vec3b col1, Vec3b col2) {
	if (col1[0] == col2[2] && col1[1] == col2[1] && col1[2] == col2[0])
		return true;
	return false;
}

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void creareImagine(){
	//creare imagine
	
	Mat img = Mat(256,256,CV_8UC3);

	//parcurgere imagine
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			// partea de sus a imaginii
			if (i < img.rows/2){
				// partea din stanga-sus
				if (j < img.cols/2)
					img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				else
					// partea din dreapta - sus
					img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			}
			else
				//partea de jos a imaginii
				//partea stanga - jos
				if (j < img.cols/2)
					img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
				else // partea dreapta - jos
					img.at<Vec3b>(i, j) = Vec3b(0, 255, 255);

		}

	}
	imshow("Imaginea", img);

	printf("Press any key to continue ...\n");
	waitKey(0);
	
}

void inversa(){
	float vals[] = { 0, 0, 15,
		10, 2, 4,
		3, 1, 7 };

	Mat m = Mat(3, 3, CV_32FC1, vals);
	
	// com.at<float>putes the inverse of a m.at<float>atrix m.at<float>
	double det = m.at<float>(0, 0) * (m.at<float>(1, 1) * m.at<float>(2, 2) - m.at<float>(2, 1) * m.at<float>(1, 2)) -
		m.at<float>(0, 1) * (m.at<float>(1, 0) * m.at<float>(2, 2) - m.at<float>(1, 2) * m.at<float>(2, 0)) +
		m.at<float>(0, 2) * (m.at<float>(1, 0) * m.at<float>(2, 1) - m.at<float>(1, 1) * m.at<float>(2, 0));

	double invdet = 1 / det;

	Mat minv = Mat(3, 3, CV_32FC1); // inverse of matrix m
	minv.at<float>(0, 0) = (m.at<float>(1, 1) * m.at<float>(2, 2) - m.at<float>(2, 1) * m.at<float>(1, 2)) * invdet;
	minv.at<float>(0, 1) = (m.at<float>(0, 2) * m.at<float>(2, 1) - m.at<float>(0, 1) * m.at<float>(2, 2)) * invdet;
	minv.at<float>(0, 2) = (m.at<float>(0, 1) * m.at<float>(1, 2) - m.at<float>(0, 2) * m.at<float>(1, 1)) * invdet;
	minv.at<float>(1, 0) = (m.at<float>(1, 2) * m.at<float>(2, 0) - m.at<float>(1, 0) * m.at<float>(2, 2)) * invdet;
	minv.at<float>(1, 1) = (m.at<float>(0, 0) * m.at<float>(2, 2) - m.at<float>(0, 2) * m.at<float>(2, 0)) * invdet;
	minv.at<float>(1, 2) = (m.at<float>(1, 0) * m.at<float>(0, 2) - m.at<float>(0, 0) * m.at<float>(1, 2)) * invdet;
	minv.at<float>(2, 0) = (m.at<float>(1, 0) * m.at<float>(2, 1) - m.at<float>(2, 0) * m.at<float>(1, 1)) * invdet;
	minv.at<float>(2, 1) = (m.at<float>(2, 0) * m.at<float>(0, 1) - m.at<float>(0, 0) * m.at<float>(2, 1)) * invdet;
	minv.at<float>(2, 2) = (m.at<float>(0, 0) * m.at<float>(1, 1) - m.at<float>(1, 0) * m.at<float>(0, 1)) * invdet;
	
	printf("Matricea dupa functia din lab:\n");
	std::cout << m.inv() << std::endl;
	printf("Matricea dupa noua functie:\n");
	std::cout << minv << std::endl;

	printf("Press any key to continue ...\n");
	int x;
	scanf("%d",&x);
}


void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);
	
	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}


void testAdaugaValoareCuloare()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int val = src.at<uchar>(i, j);
				int newVal = val + 60;
				if (newVal > 255)
					newVal = 254;
				dst.at<uchar>(i, j) = newVal;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("add image", dst);
		waitKey();
	}
}

void testMultiplicaValoareCuloare()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar newVal = val * 0.5;
				if (newVal > 255)
					newVal = 255;
				dst.at<uchar>(i, j) = newVal;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("multiplied image", dst);
		waitKey();
	}
}


void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void convertImageToRGB() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat R = Mat(height, width, CV_8UC3);
		Mat G = Mat(height, width, CV_8UC3);
		Mat B = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				B.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				G.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				R.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}

		imshow("Red", R);
		imshow("Green", G);
		imshow("Blue", B);

		waitKey(0);
	}
}

void convertImageToGrayScale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat gray = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				gray.at<uchar>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;
			}

		imshow("Gray", gray);
		waitKey(0);
	}
}

void binarizare() {
	int prag;
	printf("Prag=");
	scanf("%d", &prag);

	char fname[MAX_PATH];
	while (openFileDlg(fname)){

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat binary = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) < prag)
					binary.at<uchar>(i, j) = 0;
				else
					binary.at<uchar>(i, j) = 255;
		}

		imshow("Binary",binary);
		waitKey(0);
	}
	
}

void convertRBGtoHLS() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat H = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);

		float r, g, b;
		float M, m, C;
		
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
		
				//normalizare
				r = float(src.at<Vec3b>(i, j)[2]) / 255;
				g = float(src.at<Vec3b>(i, j)[1]) / 255;
				b = float(src.at<Vec3b>(i, j)[0]) / 255;

				M = max(max(r, g), b);
				m = min(min(r, g), b);
				C = M - m;
			
				//value
				V.at<uchar>(i, j) = M *255;


				//saturation
				if (M != 0)
					S.at<uchar>(i, j) = (C / M)*255;
				else
					S.at<uchar>(i, j) = 0;

				//hue
				float aux;
				if(C != 0) {
					if (M == r) aux = 60 * (g - b) / C;
					if (M == g) aux = 120 + 60 * (b - r) / C;
					if (M == b) aux = 240 + 60 * (r - g) / C;
				}
				else // grayscale
					aux = 0;
				if(aux < 0)
					aux += 360;
				H.at<uchar>(i, j) = aux / 360 * 255;


			}

		imshow("Hue", H);
		imshow("Saturation", S);
		imshow("Value", V);
		waitKey(0);
	}
}

void isInside() {
	char fname[MAX_PATH];
	int i, j;
	printf("Give i: ");
	scanf("%d", &i);
	printf("Give j:");
	scanf("%d", &j);
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		if (i > src.rows || i< 0)
			printf("Outside!!");
		else if (j > src.cols || j<0)
			printf("Outside!!");
		else
			printf("Inside!!");
	}
	waitKey(0);
}

int computeArea(Mat m, int red, int green, int blue) {
	int area = 0;

	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++) 
			if ((int)m.at<Vec3b>(i, j)[2] == red &&
				(int)m.at<Vec3b>(i, j)[1] == green &&
				(int)m.at<Vec3b>(i, j)[0] == blue)
				area++;

	return area;
}

void DrawCross(Mat& img, Point p, int size, Scalar color, int thickness)
{
	line(img, Point(p.x - size / 2, p.y), Point(p.x + size / 2, p.y), color, thickness, 8);
	line(img, Point(p.x, p.y - size / 2), Point(p.x, p.y + size / 2), color, thickness, 8);
}

Point calculeazaCentrulDeMasa(Mat m,int A, int red, int green, int blue ) {
	int r=0, c=0;

	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++) {
			if (m.at<Vec3b>(i, j)[0] == blue && m.at<Vec3b>(i, j)[1] == green && m.at<Vec3b>(i, j)[2] == red) {
				r += i;
				c += j;
			}
		}
	r = r / A;
	c = c / A;
	Point point = Point(c,r);

	
	return point;

}

float calculeazaAxaDeAlungire(Mat m, int c, int r, int red, int green, int blue) {

	long sumSus = 0, sumJos1 = 0, sumJos2 = 0;
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++) {
			if (m.at<Vec3b>(i, j)[0] == blue && m.at<Vec3b>(i, j)[1] == green && m.at<Vec3b>(i, j)[2] == red) {
				sumSus += ((i - r)*(j - c));
				sumJos1 += pow(j - c, 2);
				sumJos2 += pow(i - r, 2);
			}
		}

	sumSus *= 2;
	sumJos1 -= sumJos2;
	return (atan2(sumSus, sumJos1)/2.0f) ;
	
}

bool isWhite(Mat m, int i, int j) {
	
	if (m.at<Vec3b>(i - 1, j - 1)[0] == 255 && m.at<Vec3b>(i - 1, j - 1)[1] == 255 && m.at<Vec3b>(i - 1, j - 1)[2] == 255)
		return true;

	if (m.at<Vec3b>(i - 1, j)[0] == 255 && m.at<Vec3b>(i - 1, j)[1] == 255 && m.at<Vec3b>(i - 1, j)[2] == 255)
		return true;
	
	if (m.at<Vec3b>(i - 1, j + 1)[0] == 255 && m.at<Vec3b>(i - 1, j + 1)[1] == 255 && m.at<Vec3b>(i - 1, j + 1)[2] == 255)
		return true;


	if (m.at<Vec3b>(i, j - 1)[0] == 255 && m.at<Vec3b>(i, j - 1)[1] == 255 && m.at<Vec3b>(i, j - 1)[2] == 255)
		return true;

	if (m.at<Vec3b>(i, j + 1)[0] == 255 && m.at<Vec3b>(i, j + 1)[1] == 255 && m.at<Vec3b>(i, j + 1)[2] == 255)
		return true;

	if (m.at<Vec3b>(i + 1, j - 1)[0] == 255 && m.at<Vec3b>(i + 1, j - 1)[1] == 255 && m.at<Vec3b>(i + 1, j - 1)[2] == 255)
		return true;

	if (m.at<Vec3b>(i + 1, j)[0] == 255 && m.at<Vec3b>(i + 1, j)[1] == 255 && m.at<Vec3b>(i + 1, j)[2] == 255)
		return true;
	
	if (m.at<Vec3b>(i + 1, j + 1)[0] == 255 && m.at<Vec3b>(i + 1, j + 1)[1] == 255 && m.at<Vec3b>(i + 1, j + 1)[2] == 255)
		return true;

	
	return false;

}

int perimetru(Mat m, int red, int green, int blue, Mat *dst) {
	int perim = 0;
	bool ok = true;
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++) 
			if (m.at<Vec3b>(i, j)[0] == blue && m.at<Vec3b>(i, j)[1] == green && m.at<Vec3b>(i, j)[2] == red) 
				if (isWhite(m, i, j)) {
					perim++;
					dst->at<Vec3b>(i, j)[0] = 0;
					dst->at<Vec3b>(i, j)[1] = 0;
					dst->at<Vec3b>(i, j)[2] = 0;					
				}

	return perim;

}

float factorSubtiere(int A, int P) {
	return 4.0f * PI * (float(A) / (P*P));
}

float elongatia(Mat m, int red, int green, int blue) {
	int cmax = 0, cmin = m.cols, rmax = 0, rmin = m.rows;
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
			if (m.at<Vec3b>(i, j)[0] == blue && m.at<Vec3b>(i, j)[1] == green && m.at<Vec3b>(i, j)[2] == red) {
				if (i > rmax)
					rmax = i;
				if (i < rmin)
					rmin = i;
				if (j > cmax)
					cmax = j;
				if (j < cmin)
					cmin = j;
			}

	return float((cmax - cmin + 1)) / (rmax - rmin + 1);
					
}

void proiectie(Mat img, int red, int green,int blue) {
	Mat pro = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		int pixelsY = 0;
		for (int j = 0; j < img.cols; j++) {
			pro.at<uchar>(i, j) = 0;
			if (img.at<Vec3b>(i, j)[0] == blue && img.at<Vec3b>(i, j)[1] == green && img.at<Vec3b>(i, j)[2] == red) {
				pro.at<uchar>(i, pixelsY) = 255;
				pixelsY++;
			}
				
		}
	}
	
	for (int j = 0; j < img.cols; j++) {
		int pixelsX = 0;
		for (int i = 0; i < img.rows; i++)
			if (img.at<Vec3b>(i, j)[0] == blue && img.at<Vec3b>(i, j)[1] == green && img.at<Vec3b>(i, j)[2] == red) {
				pro.at<uchar>(pixelsX, j) = 255;
				pixelsX++;
			}
	}

	imshow("Proiectii", pro);

}

void onMouse(int event, int x, int y, int flags, void* param) {
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)	{
		int red = (int)(*src).at<Vec3b>(y, x)[2];
		int green = (int)(*src).at<Vec3b>(y, x)[1];
		int blue = (int)(*src).at<Vec3b>(y, x)[0];
		int A = computeArea(*src, red, green, blue);
		printf("\nArea is = %d", A);
		Point centru = calculeazaCentrulDeMasa(*src, A, red, green, blue);
		printf("\nCentru de masa : (%d,%d)", centru.x, centru.y);
		float teta = calculeazaAxaDeAlungire(*src, centru.x, centru.y, red, green, blue);
		printf("\nteta = %f", teta*180/PI);

		Mat dst = src->clone();
		int delta = 30; // arbitrary value
		Point P1, P2;
		P1.x = centru.x - delta;
		P1.y = centru.y - (int)(delta*tan(teta)); // teta is the elongation angle in radians
		P2.x = centru.x + delta;
		P2.y = centru.y + (int)(delta*tan(teta));
		line(dst, P1, P2, Scalar(0, 0, 0), 1, 8);

		
		DrawCross(dst, centru, 20, Scalar(255, 255, 255), 2);
		
		int perim = perimetru(*src, red, green, blue, &dst);
		
		printf("\nPerimetrul = %d", perim);
		printf("\nFactor de subtiere = %f", factorSubtiere(A, perim));
		printf("\nElongatia = %f", elongatia(*src, red, green, blue));
		proiectie(*src, red, green, blue);
		imshow("Cross drawn", dst);
	}
}

void computeForObject() {
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname)) {

		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", onMouse, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}

}

//used for 42
void addObjectToImage(Mat src, Mat dst, int r, int g, int b) {	
	int pixel=0;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<Vec3b>(i, j)[0] == b && src.at<Vec3b>(i, j)[1] == g && src.at<Vec3b>(i, j)[2] == r) {
				dst.at<Vec3b>(i, j)[0] = b;
				dst.at<Vec3b>(i, j)[1] = g;
				dst.at<Vec3b>(i, j)[2] = r;			
			}
}

//used for 42
bool notVerified(int v[300][3], int red, int green, int blue, int len) {
	for (int i = 0; i < len; i++)
		if (v[i][0] == blue && v[i][1] == green && v[i][2] == red)
			return false;
	return true;
}

// used for 42
void selectObjects() {

	//citeste valori
	int maxA, maxO;
	printf("Aria < ");
	scanf("%d", &maxA);
	printf("Orientarea < ");
	scanf("%d", &maxO);

	char fname[MAX_PATH];
	Mat src;
	
	int verified[300][3];
	int len = 0;	

	while (openFileDlg(fname)) {

		src = imread(fname);
		
		Vec3b color;
		int ok = 0;
		Mat d = Mat(src.rows, src.cols, CV_8UC3);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				d.at<Vec3b>(i, j)[0] = 255;
				d.at<Vec3b>(i, j)[1] = 255;
				d.at<Vec3b>(i, j)[2] = 255;
			}

		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				color = src.at<Vec3b>(i, j);
				if (!equalColor(color, Vec3b(255, 255, 255))) {

					
					if (notVerified(verified, color[2], color[1], color[0], len)) {
						verified[len][0] = color[0];
						verified[len][1] = color[1];
						verified[len][2] = color[2];
						len++;
						int A = computeArea(src, color[2], color[1], color[0]);
						Point point = calculeazaCentrulDeMasa(src, A, color[2], color[1], color[0]);
						int orient = (calculeazaAxaDeAlungire(src, point.x, point.y, color[2], color[1], color[0])) * 180 / PI;
						if (maxA > A && maxO > orient ) {
							addObjectToImage(src, d, color[2], color[1], color[0]);
						}
					}			
			
						
					}
				}
		//show the image
		imshow("My Window", d);
		
		// Wait until user press some key
		waitKey(0);
	}
}

//used for 52
void BFS(Mat src, Mat labels, bool vecinatate = false) {

	int label = 0;

	int di[8] = {-1, 0, 1, 0,-1,-1, 1, 1 };
	int dj[8] = { 0,-1, 0, 1,-1, 1,-1, 1 };
	int nLength = 4;
	if (vecinatate)
		nLength = 8;

	for (int i = 0; i < labels.rows; i++) 
		for (int j = 0; j < labels.cols; j++) 
			
			if (src.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
				std::queue<Point> que;
				label++;
				labels.at<uchar>(i, j) = label;
				que.push(Point(i, j));

				while (!que.empty()) {
					///printf("ok");
					Point oldest = que.front();
					int x = oldest.x;
					int y = oldest.y;
					que.pop();

					for (int k = 0; k < nLength; k++)

						if (src.at<uchar>(x + di[k], y + dj[k]) == 0 && labels.at<uchar>(x + di[k], y + dj[k]) == 0) {
							labels.at<uchar>(x + di[k], y + dj[k]) = label;
							que.push(Point(x + di[k], y + dj[k]));
						}


				}
			}
}

//used for 53
void etichetareCuDouaTreceri(Mat src, Mat labels, bool vecinatate = false) {

	int di[8] = { -1, 0, 1, 0,-1,-1, 1, 1 };
	int dj[8] = { 0,-1, 0, 1,-1, 1,-1, 1 };
	int nLength = 4;
	if (vecinatate)
		nLength = 8;

	int label = 0;
	std::vector<std::vector<int>> edges;
	edges.resize(255);

	for (int i = 0; i < labels.rows; i++) 
		for (int j = 0; j < labels.cols; j++) 
			if (src.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
				std::vector<int> L;

				for (int k = 0; k < nLength; k++) {
					// Neighbor coords
					int x = i + di[k];
					int y = j + dj[k];

					if (x > 0 && y > 0 && x < labels.rows && y < labels.cols) {
						if (labels.at<uchar>(x, y) > 0) {
							L.push_back(labels.at<uchar>(x, y));
						}
					}
				}

				if (L.size() == 0) {
					label++;
					labels.at<uchar>(i, j) = label;
				}
				else {
					int x = *std::min_element(L.begin(), L.end());
					//printf("min=%d", x);
					labels.at<uchar>(i, j) = x;

					for (int y : L) 
						if (y != x) {
							edges[x].push_back(y);
							edges[y].push_back(x);
							
						}
				}			
			}

	int newLabel = 0;
	std::vector<int> newLabels(label + 1, 0);

	for (int i = 1; i < label; i++) 
		if (newLabels.at(i) == 0) {
			newLabel++;
			std::queue<int> Q;
			newLabels.at(i) = newLabel;
			Q.push(i);
			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for (int y : edges[x])
					if (newLabels.at(y) == 0) {
						newLabels.at(y) = newLabel;
						Q.push(y);
					}
			}
		}

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			labels.at<uchar>(i, j) = newLabels[labels.at<uchar>(i, j)];
		}

}

// used for 52 & 53
void generareCulori(int caz) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		//generare paleta de culori
		Scalar colorLUT[1000] = { 0 };
		Scalar color;
		for (int i = 1; i < 1000; i++) {
			Scalar color(rand() & 255, rand() & 255, rand() & 255);
			colorLUT[i] = color;
		}

		colorLUT[0] = Scalar(255, 255, 255); // fundalul va fi alb

		Mat labels = Mat::zeros(src.size(), CV_16SC1); //matricea de etichete
		Mat dst = Mat::zeros(src.size(), CV_8UC3); //matricea destinatie pt. afisare

		//printf("ok");
		if (caz == 1)
			BFS(src, labels);
		else if (caz == 2)
			etichetareCuDouaTreceri(src, labels);

		printf("\nout");
		for (int i = 1; i < src.rows; i++)
			for (int j = 1; j < src.cols; j++) {
				Scalar color = colorLUT[labels.at<uchar>(i, j)]; // valabil pt. Met. 1 BFS
				dst.at<Vec3b>(i, j)[0] = color[0];
				dst.at<Vec3b>(i, j)[1] = color[1];
				dst.at<Vec3b>(i, j)[2] = color[2];
			}

		imshow("Labeled", dst);
		waitKey();
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf("10 - Image negative\n");
		printf("11 - Adauga 20 la culoare \n");
		printf("12 - Multiplica valoare culoare \n");
		printf("13 - Imagine noua\n");
		printf("14 - Calculeaza inversa\n");
		printf("\n\nLab 2:\n");
		printf("21. Image -> RGB\n");
		printf("22. Convert image to gray scale\n");
		printf("23. Binarizare\n");
		printf("24. Convert RGB to HLS\n");
		printf("25. Is Inside\n");
		printf("\n\n41. Compute for object\n");
		printf("42. Selectati obiecte\n");
		printf("\n52. Etichetare imagine");
		printf("\n53. Etichetare cu doua treceri");
		printf("\n 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1 :
				testOpenImage();
				break;
			case 2 :
				testOpenImagesFld();
				break;
			case 3 :
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4 :
				testBGR2HSV();
				break;
			case 5 :
				testResize();
				break;
			case 6 :
				testCanny();
				break;
			case 7 :
				testVideoSequence();
				break;
			case 8 :
				testSnap();
				break;
			case 9 :
				testMouseClick();
				break;
			case 10 :
				testNegativeImage();
				break;
			case 11 :
				testAdaugaValoareCuloare();
				break;
			case 12 :
				testMultiplicaValoareCuloare();
				break;
			case 13 :
				creareImagine();
				break;
			case 14 :
				inversa();
				break;
			case 21 :
				convertImageToRGB();
				break;
			case 22 :
				convertImageToGrayScale();
				break;
			case 23 :
				binarizare();
				break;
			case 24 :
				convertRBGtoHLS();
				break;
			case 25 :
				isInside();
				break;
			case 41 :
				computeForObject();
				break;
			case 42 : 
				selectObjects();
				break;
			case 52 :
				generareCulori(1);
				break;
			case 53 :
				generareCulori(2);
				break;
		}
	}
	while (op!=0);
	return 0;
}