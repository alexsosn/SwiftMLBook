
#include <opencv2/stitching.hpp>
#import "KMeansQuantizer.h"
#import <opencv2/opencv.hpp>

@implementation KMeansQuantizer

/// The class is basically an adaptation of http://stackoverflow.com/questions/14564055/opencv-k-means-clustering-in-ios

- (UIImage *)segment:(UIImage *)image {
    cv::Mat matrix = [self matrixFromImage: image];
    cv::Mat quantizedMatrix = [self kMeansClustering: matrix withK: self.k];
    UIImage *quantizedImage = [self imageFromMatrix: quantizedMatrix];
    return quantizedImage;
}

- (cv::Mat)matrixFromImage:(UIImage *) image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

- (UIImage *)imageFromMatrix:(cv::Mat)cvMat {
    NSUInteger dataLength = cvMat.elemSize()*cvMat.total();
    NSData *data = [NSData dataWithBytes:cvMat.data length:dataLength];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(
                                        cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                              //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *result = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return result;
}

- (cv::Mat)kMeansClustering:(cv::Mat)input withK:(int)k {
    cv::cvtColor(input, input, CV_RGBA2RGB);
    cv::Mat samples(input.rows * input.cols, 3, CV_32F);
    
    for (int y = 0; y < input.rows; y++){
        for (int x = 0; x < input.cols; x++){
            for (int z = 0; z < 3; z++){
                samples.at<float>(y + x*input.rows, z) = input.at<cv::Vec3b>(y,x)[z];
            }
        }
    }
    
    cv::Mat labels;
    int attempts = 1;
    cv::Mat centers;
    kmeans(samples, k, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01), attempts, cv::KMEANS_PP_CENTERS, centers);
    
    cv::Mat outputMatrix( input.rows, input.cols, input.type());
    
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            int cluster_idx = labels.at<int>(y + x*input.rows,0);
            outputMatrix.at<cv::Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
            outputMatrix.at<cv::Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
            outputMatrix.at<cv::Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }
    
    return outputMatrix;
}

@end
