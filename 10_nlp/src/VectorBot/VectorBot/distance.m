//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#import "distance.h"

static NSString * const MemoryAllocationError = @"MemoryAllocationError";

const long long maxStringLength = 2000;         // max length of strings
long long numberToShow = 40;                  // number of closest words that will be shown
const long long entryMaxLength = 50;              // max length of vocabulary entries

/*
 Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT
 */

@interface W2VDistance () {
    char file_name[maxStringLength];
    char *vocab;
    float *M;
}

@property (nonatomic) FILE *modelFile;
@property (nonatomic, strong) NSNumber *wordsTotalNum; // long long
@property (nonatomic, strong) NSNumber *size; // long long

@end

@implementation W2VDistance

- (void)loadBinaryVectorFile:(NSURL * _Nonnull) fileURL
                       error:(NSError ** _Nullable) error {

    NSFileManager *manager = [NSFileManager defaultManager];
    char const *outputFilePath = [manager fileSystemRepresentationWithPath:fileURL.path];
    
    strcpy(file_name, outputFilePath);
    self.modelFile = fopen(file_name, "rb");
    
    long long wordsTotalNum, size;

    fscanf(self.modelFile, "%lld", &wordsTotalNum);
    fscanf(self.modelFile, "%lld", &size);
    
    self.wordsTotalNum = @(wordsTotalNum);
    self.size = @(size);
    

    vocab = (char *)malloc((long long)wordsTotalNum * entryMaxLength * sizeof(char));

    M = (float *)malloc((long long)wordsTotalNum * (long long)size * sizeof(float));
    
    if (M == NULL) {
        NSString *message = [NSString stringWithFormat:@"Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)wordsTotalNum * size * sizeof(float) / 1048576, wordsTotalNum, size];
        
        * error = [NSError errorWithDomain:MemoryAllocationError
                                      code:1
                                  userInfo:@{NSLocalizedDescriptionKey : message}];
        
        NSLog(@"%@", message);
        return;
    }
    
    long long a, b;
    float len;

    for (b = 0; b < wordsTotalNum; b++) {
        a = 0;
        while (1) {
            vocab[b * entryMaxLength + a] = fgetc(self.modelFile);
            if (feof(self.modelFile) || (vocab[b * entryMaxLength + a] == ' ')) break;
            if ((a < entryMaxLength) && (vocab[b * entryMaxLength + a] != '\n')) a++;
        }
        vocab[b * entryMaxLength + a] = 0;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, self.modelFile);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(self.modelFile);
}

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)closestToWord:(NSString * _Nonnull) word
                                                    numberOfClosest:(NSNumber * _Nullable) numberOfClosest {
    
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    
    if(numberOfClosest) { numberToShow = numberOfClosest.longLongValue; }
    
    char st1[maxStringLength];
    char *bestWords[numberToShow];
    char st[100][maxStringLength];
    float dist, len, bestDistances[numberToShow], vec[maxStringLength];
    long long a, b, c, d, cn, bi[100];
    
    for (a = 0; a < numberToShow; a++) {
        bestWords[a] = (char *)malloc(maxStringLength * sizeof(char));
    }
    
    for (a = 0; a < numberToShow; a++) bestDistances[a] = 0;
    for (a = 0; a < numberToShow; a++) bestWords[a][0] = 0;
    a = 0;
    while (1) {
        st1[a] = [word cStringUsingEncoding:NSUTF8StringEncoding][a];
        if ((st1[a] == '\n') || (a >= maxStringLength - 1)) {
            st1[a] = 0;
            break;
        }
        a++;
    }
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
        st[cn][b] = st1[c];
        b++;
        c++;
        st[cn][b] = 0;
        if (st1[c] == 0) break;
        if (st1[c] == ' ') {
            cn++;
            b = 0;
            c++;
        }
    }
    cn++;

    long long wordsTotalNum = self.wordsTotalNum.longLongValue;

    for (a = 0; a < cn; a++) {
        for (b = 0; b < wordsTotalNum; b++) {
            if (!strcmp(&vocab[b * entryMaxLength], st[a])) {
                break;
            }
        }
        if (b == wordsTotalNum) b = -1;
        bi[a] = b;
        
        if (b == -1) {
            NSLog(@"Out of dictionary word!\n");
            break;
        }
    }
    
    long long size = self.size.longLongValue;
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
        if (bi[b] == -1) continue;
        for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < numberToShow; a++) bestDistances[a] = -1;
    for (a = 0; a < numberToShow; a++) bestWords[a][0] = 0;
    for (c = 0; c < wordsTotalNum; c++) {
        a = 0;
        for (b = 0; b < cn; b++) {
            if (bi[b] == c) {
                a = 1;
            }
        }
        if (a == 1) continue;
        dist = 0;
        for (a = 0; a < size; a++) {
            dist += vec[a] * M[a + c * size];
        }
        for (a = 0; a < numberToShow; a++) {
            if (dist > bestDistances[a]) {
                for (d = numberToShow - 1; d > a; d--) {
                    bestDistances[d] = bestDistances[d - 1];
                    strcpy(bestWords[d], bestWords[d - 1]);
                }
                bestDistances[a] = dist;
                strcpy(bestWords[a], &vocab[c * entryMaxLength]);
                break;
            }
        }
    }
    
    for (a = 0; a < numberToShow; a++) {
        result[[NSString stringWithCString:bestWords[a] encoding:NSUTF8StringEncoding]] = @(bestDistances[a]);
    }
    
    free(*bestWords);
    return result;
}

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)analogyToPhrase:(NSString * _Nonnull) phrase
                                                    numberOfClosest:(NSNumber * _Nullable) numberOfClosest {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if(numberOfClosest) { numberToShow = numberOfClosest.longLongValue; }
    char st1[maxStringLength];
    char *bestWords[numberToShow];
    char st[100][maxStringLength];
    float dist, len, bestDistances[numberToShow], vec[maxStringLength];
    long long a, b, c, d, cn, bi[100];
    
    for (a = 0; a < numberToShow; a++) {
        bestWords[a] = (char *)malloc(maxStringLength * sizeof(char));
    }
    for (a = 0; a < numberToShow; a++) bestDistances[a] = 0;
    for (a = 0; a < numberToShow; a++) bestWords[a][0] = 0;
    a = 0;
    while (1) {
        st1[a] = [phrase cStringUsingEncoding:NSUTF8StringEncoding][a];
        if ((st1[a] == '\n') || (a >= maxStringLength - 1)) {
            st1[a] = 0;
            break;
        }
        a++;
    }
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
        st[cn][b] = st1[c];
        b++;
        c++;
        st[cn][b] = 0;
        if (st1[c] == 0) break;
        if (st1[c] == ' ') {
            cn++;
            b = 0;
            c++;
        }
    }
    cn++;
    
    long long wordsTotalNum = self.wordsTotalNum.longLongValue;

    for (a = 0; a < cn; a++) {
        for (b = 0; b < wordsTotalNum; b++) if (!strcmp(&vocab[b * entryMaxLength], st[a])) break;
        if (b == wordsTotalNum) b = 0;
        bi[a] = b;
        if (b == 0) {
            NSLog(@"Out of dictionary word!\n");
            break;
        }
    }

    long long size = self.size.longLongValue;

    NSInteger wordNumber = [phrase componentsSeparatedByString:@" "].count;
    if (wordNumber == 2) {
        for (a = 0; a < size; a++) vec[a] = M[a + bi[0] * size] - M[a + bi[1] * size];
    } else {
        for (a = 0; a < size; a++) vec[a] = M[a + bi[1] * size] - M[a + bi[0] * size] + M[a + bi[2] * size];
    }
    
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < numberToShow; a++) bestDistances[a] = 0;
    for (a = 0; a < numberToShow; a++) bestWords[a][0] = 0;
    for (c = 0; c < wordsTotalNum; c++) {
        if (c == bi[0]) continue;
        if (c == bi[1]) continue;
        if (c == bi[2]) continue;
        a = 0;
        for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
        if (a == 1) continue;
        dist = 0;
        for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
        for (a = 0; a < numberToShow; a++) {
            if (dist > bestDistances[a]) {
                for (d = numberToShow - 1; d > a; d--) {
                    bestDistances[d] = bestDistances[d - 1];
                    strcpy(bestWords[d], bestWords[d - 1]);
                }
                bestDistances[a] = dist;
                strcpy(bestWords[a], &vocab[c * entryMaxLength]);
                break;
            }
        }
    }
    
    for (a = 0; a < numberToShow; a++) {
        result[[NSString stringWithCString:bestWords[a] encoding:NSUTF8StringEncoding]] = @(bestDistances[a]);
    }
    
    free(*bestWords);
    return result;
}

- (void)dealloc
{
    free(vocab);
    free(M);
}

@end

