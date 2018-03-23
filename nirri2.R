#Data Capimc
DataCS=data.frame(result=c(56359,56276,57840,57260,78642,78206,77252,78414,69684,69972,68798,68693,58411,58047,57282,58316),
treatment=c(rep("0",4),rep("50",4),rep("100",4),rep("150",4)), block=c(1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4))
str(DataCS)
DataCS

DataCS$block <- as.factor(DataCS$block)
str(DataCS)

#Test ANOVA
analysis <- aov(result~ treatment + block, DataCS)
anova(analysis)
summary(anova)

#install.packages("agricolae")
library("agricolae")
#Test of Tukey
tukey <- HSD.test(analysis,"treatment", alpha=0.05)
tukey

#install.packages("car")
library(car)
#Test for Homogeneity
levene.test(result~treatment,data=DataCS)

#Shapiro-Wilk normality test
shapiro.test(residuals(analysis))

#GoogleVis
install.packages("googleVis")
library(googleVis)
df.data <- data.frame(DataCS$result,DataCS$treatment)
mychart <- gvisLineChart(df.data, options=list(gvis.editor="edit",width=1000,height=600))
plot(mychart)

tukey$groups %>% 
  rownames_to_column(var = "trt") %>% 
  ggplot(aes(reorder(trt, DataCS$result, function(x) -mean(x)), DataCS$result)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = groups), vjust = 1.8, size = 9, color = "white") +
    labs(x = "Progênies", y = "Médias") +
    theme_few()