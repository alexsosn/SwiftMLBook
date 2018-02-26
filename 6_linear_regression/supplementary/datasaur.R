setwd("~/projects/personal/book_swift_ml_code/03Regression/figures")

library(readr)
DatasaurusDozen <- read_delim("DatasaurusDozen.tsv", 
                              "\t", escape_double = FALSE, trim_ws = TRUE)


DatasaurusDozen$dataset=as.factor(DatasaurusDozen$dataset)
lvls = c("bullseye", "circle", "high_lines",  "wide_lines",
         "h_lines", "v_lines", "slant_down", "slant_up", 
          "dots", "star", "x_shape", "dino")

png("datasaurus.png", width = 5.5, height = 5.5, units = "in", res = 300)

par(mfrow=c(3,4),
    mai=c(0.1,0.1,0.3,0.1), 
    oma=c(0.3,0.3,0.3,0.3), lab=c(2,2,7), 
    mgp = c(0, 1, 0))
# by(DatasaurusDozen[,2:3], DatasaurusDozen$dataset, plot)


for(l in lvls) {
  x = DatasaurusDozen[DatasaurusDozen$dataset==l,]$x
  y = DatasaurusDozen[DatasaurusDozen$dataset==l,]$y
  params = lm(y ~ x)$coefficients
  b = params[1]
  a = params[2]
  main = paste("a=",round(a, digits=2),"\nb=",round(b, digits=2))
  plot(x,y, 
       xlim = c(0,100), 
       ylim = c(0,100), 
       ylab="", 
       xlab = "", 
       col = "gray", 
       pch = 21, 
       bg = "gray", 
       cex = 0.5, 
       main = main, 
       asp = c(1,1), 
       xaxt='n', 
       yaxt='n')
  abline(lm(y ~ x), col = "blue")
}

dev.off()

