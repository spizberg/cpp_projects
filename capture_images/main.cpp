#include <iostream>
#include <opencv2/opencv.hpp>
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>

int main() {
    Pylon::PylonAutoInitTerm autoInitTerm;
    try {
        Pylon::CBaslerUniversalInstantCamera camera( Pylon::CTlFactory::GetInstance().CreateFirstDevice() );
        // Print the model name of the camera.
        std::cout << "Using device " << camera.GetDeviceInfo().GetModelName() << std::endl;

        camera.Open();
        // camera.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly);
        camera.ExposureTime.SetValue(10000);
        camera.StartGrabbing(Pylon::GrabStrategy_LatestImages);
        Pylon::CImageFormatConverter converter {Pylon::CImageFormatConverter()};
        converter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
        converter.OutputBitAlignment = Pylon::OutputBitAlignment_MsbAligned;
        Pylon::CGrabResultPtr grabResultPtr;
        cv::Mat opencvImage;
        Pylon::CPylonImage pylonImage;
        cv::namedWindow("Basler Stream");
        cv::moveWindow("Basler Stream", 320, 28);
        int counter{1};
        std::ostringstream s;

        while (camera.IsGrabbing()){
            camera.RetrieveResult(5000, grabResultPtr, Pylon::TimeoutHandling_ThrowException);

            if (grabResultPtr->GrabSucceeded()){
                converter.Convert(pylonImage, grabResultPtr);
                opencvImage = cv::Mat(grabResultPtr->GetHeight(), grabResultPtr->GetWidth(), CV_8UC3, (uint8_t *)pylonImage.GetBuffer());

                cv::imshow("Basler Stream", opencvImage);
                char c = cv::waitKey(33);
                if (c == 'q') {
                    cv::destroyAllWindows();
                    break;
                }
                if (c == 'c') {
                    s << "test_basler_" << counter << ".png";
                    cv::imwrite("/home/nathan/Images/seg_test/" + s.str(), opencvImage,
                                std::vector<int>{cv::IMWRITE_PNG_COMPRESSION, 0});
                    counter++;
                    s.clear();
                    s.str("");
                }
            }
            grabResultPtr.Release();
        }
        camera.StopGrabbing();
    }
    catch (const GenICam_3_1_Basler_pylon::GenericException& e){

    }

    return 0;
}
