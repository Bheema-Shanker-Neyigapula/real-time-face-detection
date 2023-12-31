#include <opencv2/opencv.hpp>

int main() {
    // Open the default camera
    cv::VideoCapture cap(0);

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error opening the camera" << std::endl;
        return -1;
    }

    // Load pre-trained face cascade
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade" << std::endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Face Detection", cv::WINDOW_NORMAL);

    while (true) {
        // Read a frame from the camera
        cv::Mat frame;
        cap >> frame;

        // Convert the frame to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect faces in the frame
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.3, 5);

        // Draw rectangles around the faces
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }

        // Display the frame
        cv::imshow("Face Detection", frame);

        // Break the loop if the user presses the 'Esc' key
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Release the camera and close the window
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
