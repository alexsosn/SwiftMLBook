png("anscombe.png", width = 5.5, height = 5.5, units = "in", res = 300)

par(mfrow=c(2,2),
    mai=c(0.5,0.5,0.5,0.5),
    oma=c(0.75,0.75,0.75,0.75),
    lab=c(2,2,7),
    mgp = c(0, 1, 0)
    )

require(stats); require(graphics)
# summary(anscombe)

##-- now some "magic" to do the 4 regressions in a loop:
ff <- y ~ x
mods <- setNames(as.list(1:4), paste0("lm", 1:4))
for(i in 1:4) {
  ff[2:3] <- lapply(paste0(c("y","x"), i), as.name)
  mods[[i]] <- lmi <- lm(ff, data = anscombe)
  # print(anova(lmi))
}

## See how close they are (numerically!)
sapply(mods, coef)
lapply(mods, function(fm) coef(summary(fm)))

## Now, do what you should have done in the first place: PLOTS
op <- par(mfrow = c(2, 2), mar = 0.1+c(4,4,1,1), oma =  c(0, 0, 2, 0))
for(i in 1:4) {
  ff[2:3] <- lapply(paste0(c("y","x"), i), as.name)
  plot(ff, data = anscombe, 
       col = "gray", 
       pch = 21, 
       bg = "gray", 
       cex = 1.5,
       xlim = c(0, 20), ylim = c(0, 20),
       xlab = "",
       ylab = ""
       )
  abline(mods[[i]], col = "red")
}
# mtext("Anscombe's 4 Regression data sets", outer = TRUE, cex = 1.5)
par(op)

dev.off()
