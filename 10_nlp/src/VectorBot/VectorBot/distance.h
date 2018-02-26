
#import <Foundation/Foundation.h>

@interface W2VDistance : NSObject

- (void)loadBinaryVectorFile:(NSURL * _Nonnull) fileURL
                       error:(NSError *_Nullable* _Nullable) error;

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)closestToWord:(NSString * _Nonnull) word
                                                    numberOfClosest:(NSNumber * _Nullable) numberOfClosest;

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)analogyToPhrase:(NSString * _Nonnull) phrase
                                                      numberOfClosest:(NSNumber * _Nullable) numberOfClosest;

@end
