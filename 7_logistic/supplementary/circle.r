hours = c(23,1,10,17)
mins = c(45,15,55,30)
labels = c("black","black","gray","gray")
labs = c(1,1,0,0)

sins = sin(pi*2*(hours*60+mins)/(24*60))
coss = cos(pi*2*(hours*60+mins)/(24*60))

data = data.frame(sin=sins, cos=coss, labels=labels)

png("circle_abline.png", width = 5.5, height = 5.5, units = "in", res = 300)

plot(sins,coss, asp=1, col=labels, 
     xlim=c(-1,1), ylim=c(-1,1),
     xlab="cos", ylab = "sin")
abline(v=0)
abline(h=0)
symbols(x=0, y=0, circles=c(1), add=T, inches=F)

g=glm(labs~coss+sins,family=binomial)

slope <- coef(g)[2]/(-coef(g)[3])
intercept <- coef(g)[1]/(-coef(g)[3]) 
abline(intercept, slope)

dev.off()

png("min_hour.png", width = 5.5, height = 5.5, units = "in", res = 300)

plot(hours, mins, col=labels, 
     xlim=c(0, 25), ylim=c(0, 59),
     xlab="hours", ylab = "minutes")

dev.off()
