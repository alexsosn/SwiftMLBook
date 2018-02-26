png("errors.png", width = 5.5, height = 5.5, units = "in", res = 300)

set.seed(123)
par(mfrow=c(2,2),
    mai=c(0.1,0.1,0.3,0.1), 
    oma=c(0.3,0.3,0.3,0.3), lab=c(2,2,7), 
    mgp = c(0, 1, 0))

x = runif(100, 0, 100)
y = 0.75*x + 5 + rnorm(100, 0, 10)
plot(x,y,
     xlim = c(0,100), 
     ylim = c(0,100), 
     ylab="", 
     xlab = "", 
     col = "gray", 
     pch = 21, 
     bg = "gray", 
     cex = 0.5, 
     asp = c(1,1), 
     xaxt='n', 
     yaxt='n',
     main = 'Good')

abline(lm(y~x), col="red")

y = 0.75*x + 5 + rnorm(100, 0, 5)*0.03*x
plot(x,y,
     xlim = c(0,100), 
     ylim = c(0,100), 
     ylab="", 
     xlab = "", 
     col = "gray", 
     pch = 21, 
     bg = "gray", 
     cex = 0.5, 
     asp = c(1,1), 
     xaxt='n', 
     yaxt='n',
     main = 'Heteroscedasticity')

abline(lm(y~x), col="red")

y = 0.75*x + 5 + runif(100, 0, 15)*cos(x/5)
plot(x,y,
     xlim = c(0,100), 
     ylim = c(0,100), 
     ylab="", 
     xlab = "", 
     col = "gray", 
     pch = 21, 
     bg = "gray", 
     cex = 0.5, 
     asp = c(1,1), 
     xaxt='n', 
     yaxt='n',
     main = 'Errors are not independent')

abline(lm(y~x), col="red")

mu1 <- log(10)   
mu2 <- log(10)
sig1 <- log(2)
sig2 <- log(5)
cpct <- 0.5  

bimodalDistFunc <- function (n,cpct, mu1, mu2, sig1, sig2) {
  y0 <- rlnorm(n,mean=mu1, sd = sig1)
  y1 <- rlnorm(n,mean=mu2, sd = sig2)
  
  flag <- rbinom(n,size=1,prob=cpct)
  y <- y0*(1 - flag) + y1*flag 
}

bimodalData <- bimodalDistFunc(n=100,cpct,mu1,mu2, sig1,sig2)

y = 0.75*x + bimodalData -20
plot(x,y,
     xlim = c(0,100), 
     ylim = c(0,100), 
     ylab="", 
     xlab = "", 
     col = "gray", 
     pch = 21, 
     bg = "gray", 
     cex = 0.5, 
     asp = c(1,1), 
     xaxt='n', 
     yaxt='n',
     main = "Error distribution is not normal")

abline(lm(y~x), col="red")

dev.off()
