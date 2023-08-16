library(FramebyFrame)
wd<-"S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/"
setwd(wd)

vpSorter(ffDir="S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/220815_14_15_Gria3Trio/220815_14_15_rawoutput",
         zebpath="S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/220815_14_15_Gria3Trio/220815_14_15_Gria3Trio.xls",
         boxGen=2,
         twoBoxMode=TRUE,
         boxnum=1,
         zt0="09:00:00",
         dayduration=14)

vpSorter(ffDir="S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/220815_14_15_Gria3Trio/220815_14_15_rawoutput",
         zebpath="S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/220815_14_15_Gria3Trio/220815_14_15_Gria3Trio.xls",
         boxGen=2,
         twoBoxMode=TRUE,
         boxnum=2,
         zt0="09:00:00",
         dayduration=14)