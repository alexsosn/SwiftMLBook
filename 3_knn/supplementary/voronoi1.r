## HF#2 Metrics: Euclidean, Manhattan and Minkovski
Metric <- function(x, y, mt) {
  if(mt==1) {return(sqrt(x*x + y*y))}
  if(mt==2) {return(abs(x) + abs(y))}
  if(mt==3) {return((abs(x)^3 + abs(y)^3)^0.33333)}
}

## Plotting Voronoi diagram. aev 3/12/17
## ns - number of sites, fn - file name, ttl - plot title.
## mt - type of metric: 1 - Euclidean, 2 - Manhattan, 3 - Minkovski.
pVoronoiD <- function(ns, fn="", ttl="",mt=1) {
  set.seed(15)
  
  cat(" *** START VD:", date(), "\n");
  if(mt<1||mt>3) {mt=1}; mts=""; if(mt>1) {mts=paste0(", mt - ",mt)}; 
  m=640; i=j=k=m1=m-2; x=y=d=dm=0;
  if(fn=="") {pf=paste0("VDR", mt, ns, ".png")} else {pf=paste0(fn, ".png")};
  if(ttl=="") {ttl=paste0("Voronoi diagram, sites - ", ns, mts)};
  cat(" *** Plot file -", pf, "title:", ttl, "\n");
  plot(NA, xlim=c(0,m), ylim=c(0,m), xlab="", ylab="", main=ttl);
  X=numeric(ns); Y=numeric(ns); C=numeric(ns);
  cols = rainbow(ns) #gray.colors(ns)
  for(i in 1:ns) {
    X[i]=sample(0:m1, 1, replace=TRUE); 
    Y[i]=sample(0:m1, 1, replace=TRUE);
    C[i]=cols[i]
  }
  for(i in 0:m1) {
    for(j in 0:m1) {
      dm=Metric(m1,m1,mt); k=-1;
      for(n in 1:ns) {
        d=Metric(X[n]-j,Y[n]-i, mt);
        if(d<dm) {dm=d; k=n;}
      }
      clr=C[k]; segments(j, i, j, i, col=clr);
    }
  }
  points(X, Y, pch = 19, col = "black", bg = "white")
  dev.copy(png, filename=pf, width=m, height=m);
  dev.off(); graphics.off();
  cat(" *** END VD:",date(),"\n");
}
## Executing:
pVoronoiD(15)          ## Euclidean metric
pVoronoiD(15,"","",2)   ## Manhattan metric
pVoronoiD(15,"","",3)   ## Minkovski metric
