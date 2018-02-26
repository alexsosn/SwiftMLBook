

# Random sample in the unit square
x <- matrix(runif(100), nc = 2)
# Delaunay triangulation and Voronoi diagram
delvor.obj <- delvor(x)
# Plot Voronoi diagram and Delaunay triangulation 
plot(delvor.obj, wlines = "vor", col = c(1,2,3,4))

set.seed(42)
x <- runif(20)
y <- runif(20)
z <- deldir(x,y,rw=c(0,1,0,1))
w <- tile.list(z)

ccc <- terrain.colors(20)
plot(w,fillcol=ccc,close=TRUE)

