step = function(x) { as.integer(x>=0) }
logistic = function(x) { 1/(1+exp(-x)) }
relu = function(x) { ifelse(x>=0,x,0) }
leakyrelu = function(x) { ifelse(x>=0,x,0.05*x) }
softplus = function(x) { log(1+exp(x)) }

png("activation.png", width = 5.5, height = 5.5, units = "in", res = 300)

par(mfrow=c(3,2), mai=c(0.3,0.3,0.3,0.3), 
    oma=c(0.3,0.3,0.3,0.3), lab=c(2,2,7), 
    mgp = c(0, 1, 0))

curve(step, from=-5, to=5, ylab="", xlab = "", main = "Step", ylim=c(-1,1), lwd=3)
abline(h=0)
abline(v=0)
curve(relu, from=-5, to=5, ylab="", xlab= "", main = "ReLU", ylim=c(-1,1), lwd=3)
abline(h=0)
abline(v=0)

curve(logistic, from=-5, to=5, ylab="", xlab= "", main = "Logistic", ylim=c(-1,1), lwd=3)
abline(h=0)
abline(v=0)
curve(leakyrelu, from=-5, to=5, ylab="", xlab= "", main = "Leaky ReLU, a=0.05", ylim=c(-1,1), lwd=3)
abline(h=0)
abline(v=0)

curve(tanh, from=-5, to=5, ylab="", xlab= "", main = "Hyperbolic tangent", ylim=c(-1,1), lwd=3)
abline(h=0)
abline(v=0)
curve(softplus, from=-5, to=5, ylab="", xlab= "", main = "Softplus", ylim=c(-1,1), lwd=3)
abline(h=0)
abline(v=0)

dev.off()
