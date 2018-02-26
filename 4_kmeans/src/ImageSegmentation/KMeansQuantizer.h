
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface KMeansQuantizer : NSObject

@property int k;

- (UIImage *)segment:(UIImage *)image;

@end
