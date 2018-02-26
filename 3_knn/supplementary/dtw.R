walk = read.csv("~/Downloads/HMP_Dataset/Walk/Accelerometer-2011-03-24-09-51-07-walk-f1.txt",
                sep=' ', header = F)

walk2 = read.csv("~/Downloads/HMP_Dataset/Walk/Accelerometer-2011-03-24-09-52-11-walk-f1.txt",
                sep=' ', header = F)

brush = read.csv("~/Downloads/HMP_Dataset/Brush_teeth/Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt",
                 sep=' ', header = F)


s1 = sqrt(walk$V1^2+walk$V2^2+walk$V3^2)[200:300]
s2 = sqrt(walk2$V1^2+walk2$V2^2+walk2$V3^2)[200:300]
s3 = sqrt(brush$V1^2+brush$V2^2+brush$V3^2)[200:300]

## Find the best match with the canonical recursion formula
library(dtw);

## Display the warping curve, i.e. the alignment curve
png("dtw_4plots.png", width = 5.5, height = 5.5, units = "in", res = 300)

par(mfrow=c(2,2))

alignment<-dtw(s2,s1,keep=TRUE);
plot(alignment,type="threeway", xlab='Walk', ylab='Walk', main=NULL)

alignment<-dtw(s3,s1,keep=TRUE);
plot(alignment,type="threeway", ylab='Walk', xlab='Brush teeth', main=NULL)

plot(
  dtw(s1,s2,keep=TRUE,
      step=rabinerJuangStepPattern(6,"c")),
  offset=-10, xlab='Time', ylab='Acceleration magnitude',
  type="twoway")

plot(
  dtw(s1,s3,keep=TRUE,
      step=rabinerJuangStepPattern(6,"c")),
  type="twoway",offset=-50, xlab='Time', ylab='Acceleration magnitude')
dev.off()

############

png("dtw_walk.png", width = 5.5, height = 5.5, units = "in", res = 300)

par(mfrow=c(2,2), mai=c(0.3,0.3,0.3,0.3), 
    oma=c(0.3,0.3,0.3,0.3), lab=c(2,2,7), 
    mgp = c(0, 1, 0))

alignment<-dtw(s2,s1,keep=TRUE);
plot(alignment,type="threeway", xlab='Walk', ylab='Walk', main=NULL)

png("dtw_walk_brush.png", width = 5.5, height = 5.5, units = "in", res = 300)
alignment<-dtw(s3,s1,keep=TRUE);
plot(alignment,type="threeway", ylab='Walk', xlab='Brush teeth', main=NULL)
dev.off()

png("spring_walk_walk.png", width = 5.5, height = 5.5, units = "in", res = 300)
plot(
  dtw(s1,s2,keep=TRUE,
      step=rabinerJuangStepPattern(6,"c")),
  offset=-10, xlab='Time', ylab='Acceleration magnitude',
  type="twoway")
dev.off()

png("spring_walk_brush.png", width = 5.5, height = 5.5, units = "in", res = 300)
plot(
  dtw(s1,s3,keep=TRUE,
      step=rabinerJuangStepPattern(6,"c")),
  type="twoway",offset=0, xlab='Time', ylab='Acceleration magnitude')
dev.off()

