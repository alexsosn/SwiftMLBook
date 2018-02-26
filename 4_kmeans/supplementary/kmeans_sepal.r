require(graphics)
require(deldir)

png("kmeans_bw_voronoi.png", width = 5.5, height = 5.5, units = "in", res = 300)

par(mfrow=c(2,2), mai=c(0.2,0.2,0.2,0.2), xaxt="n", yaxt="n")

pch = c(3,21,22,23,24)

cl = kmeans(as.matrix(iris[,1:2]), centers=2)
vtess <- deldir(cl$centers[,1], cl$centers[,2], rw=c(4,8,1,8))
plot(iris$Sepal.Length, iris$Sepal.Width, type="n", asp=1, main = 'k=2', ylab = 'Sepal Width', xlab='Sepal Length')
points(iris$Sepal.Length, iris$Sepal.Width, bg = gray.colors(2)[cl$cluster], pch=pch[cl$cluster])
plot(vtess, wlines="tess", wpoints="none", number=FALSE, add=TRUE, lty=1)
points(cl$centers, pch=19, cex=1.5, font=2)

cl = kmeans(as.matrix(iris[,1:2]), centers=3)
vtess <- deldir(cl$centers[,1], cl$centers[,2], rw=c(4,8,1,8))
plot(iris$Sepal.Length, iris$Sepal.Width, type="n", asp=1, main = 'k=3', ylab = 'Sepal Width', xlab='Sepal Length')
points(iris$Sepal.Length, iris$Sepal.Width, bg = gray.colors(3)[cl$cluster], pch=pch[cl$cluster])
plot(vtess, wlines="tess", wpoints="none", number=FALSE, add=TRUE, lty=1)
points(cl$centers, pch=19, cex=1.5, font=2)

cl = kmeans(as.matrix(iris[,1:2]), centers=4)
vtess <- deldir(cl$centers[,1], cl$centers[,2], rw=c(4,8,1,8))
plot(iris$Sepal.Length, iris$Sepal.Width, type="n", asp=1, main = 'k=4', ylab = 'Sepal Width', xlab='Sepal Length')
points(iris$Sepal.Length, iris$Sepal.Width, bg = gray.colors(4)[cl$cluster], pch=pch[cl$cluster])
plot(vtess, wlines="tess", wpoints="none", number=FALSE, add=TRUE, lty=1)
points(cl$centers, pch=19, cex=1.5, font=2)

cl = kmeans(as.matrix(iris[,1:2]), centers=5)
vtess <- deldir(cl$centers[,1], cl$centers[,2], rw=c(4,8,1,8))
plot(iris$Sepal.Length, iris$Sepal.Width, type="n", asp=1, main = 'k=5', ylab = 'Sepal Width', xlab='Sepal Length')
points(iris$Sepal.Length, iris$Sepal.Width, bg = gray.colors(5)[cl$cluster], pch=pch[cl$cluster])
plot(vtess, wlines="tess", wpoints="none", number=FALSE, add=TRUE, lty=1)
points(cl$centers, pch=19, cex=1.5, font=2)

dev.off()

