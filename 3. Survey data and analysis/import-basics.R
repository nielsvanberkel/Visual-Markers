using<-function(...) {
  libs<-unlist(list(...))
  req<-unlist(lapply(libs,require,character.only=TRUE))
  need<-libs[req==FALSE]
  n<-length(need)
  if(n>0){
    libsmsg<-if(n>2) paste(paste(need[1:(n-1)],collapse=", "),",",sep="") else need[1]
    print(libsmsg)
    if(n>1){
      libsmsg<-paste(libsmsg," and ", need[n],sep="")
    }
    libsmsg<-paste("The following packages could not be found: ",libsmsg,"\n\r\n\rInstall missing packages?",collapse="")
    if(winDialog(type = c("yesno"), libsmsg)=="YES"){       
      install.packages(need)
      lapply(need,require,character.only=TRUE)
    }
  }
}



##### Setup #####

# Loading the required libraries using the above function. Alternatively, install.packages("package_name") + library(package_name)
using("tidyverse","reshape2","ggpubr","xtable", "ARTool", "emmeans", "PMCMRplus")

survey_data <- read.csv("survey_data.csv")

str(survey_data)



##### Structure data #####

# reverse code rank questions (nr 1 gets 7 points, nr 2 gets 6 points, etc)
survey_data <- survey_data %>% 
  mutate_at(.funs = funs(car::recode(., "7=1;6=2;5=3;4=4;3=5;2=6;1=7")), .vars = c("RankWideBoundingCircle", 
                                                                               "RankTightBoundingBox", "RankSpotlight", 
                                                                               "RankSegmentation", "RankSegmentationOutline",
                                                                               "RankDetectionConfidence", "RankDetectionSignal"))
# reverse code colour preferences
survey_data <- survey_data %>% 
  mutate_at(.funs = funs(car::recode(., "7=1;6=2;5=3;4=4;3=5;2=6;1=7")), .vars = c("Black", "Blue", "Green", "Orange", "Purple", "Red", "Yellow"))

str(survey_data$Response.ID)
survey_data %>% select(Response.ID)
likert <- survey_data %>% select(Response.ID, 
                                 Easy_DetectMorePolyps0, Easy_FasterPolyps0, Easy_InterferePolypectomy0, Easy_InterfereRegular0,
                                 Easy_DetectMorePolyps1, Easy_FasterPolyps1, Easy_InterferePolypectomy1, Easy_InterfereRegular1,
                                 Easy_DetectMorePolyps2, Easy_FasterPolyps2, Easy_InterferePolypectomy2, Easy_InterfereRegular2,
                                 Easy_DetectMorePolyps3, Easy_FasterPolyps3, Easy_InterferePolypectomy3, Easy_InterfereRegular3,
                                 Easy_DetectMorePolyps4, Easy_FasterPolyps4, Easy_InterferePolypectomy4, Easy_InterfereRegular4,
                                 Easy_DetectMorePolyps5, Easy_FasterPolyps5, Easy_InterferePolypectomy5, Easy_InterfereRegular5,
                                 Easy_DetectMorePolyps6, Easy_FasterPolyps6, Easy_InterferePolypectomy6, Easy_InterfereRegular6,
                                 Difficult_DetectMorePolyps0, Difficult_FasterPolyps0, Difficult_InterferePolypectomy0, Difficult_InterfereRegular0,
                                 Difficult_DetectMorePolyps1, Difficult_FasterPolyps1, Difficult_InterferePolypectomy1, Difficult_InterfereRegular1,
                                 Difficult_DetectMorePolyps2, Difficult_FasterPolyps2, Difficult_InterferePolypectomy2, Difficult_InterfereRegular2,
                                 Difficult_DetectMorePolyps3, Difficult_FasterPolyps3, Difficult_InterferePolypectomy3, Difficult_InterfereRegular3,
                                 Difficult_DetectMorePolyps4, Difficult_FasterPolyps4, Difficult_InterferePolypectomy4, Difficult_InterfereRegular4,
                                 Difficult_DetectMorePolyps5, Difficult_FasterPolyps5, Difficult_InterferePolypectomy5, Difficult_InterfereRegular5,
                                 Difficult_DetectMorePolyps6, Difficult_FasterPolyps6, Difficult_InterferePolypectomy6, Difficult_InterfereRegular6) %>%
  melt(id.vars = c("Response.ID")) %>%
  separate(variable, c("Video", "variable"))

# Likert
likert <- likert %>%
  separate(col = "variable", into = c("Variable", "Design"), sep = -1) %>% 
  spread("Variable", value)

str(likert$Design)
likert$Design <- factor(likert$Design, labels=c("I. Wide bounding circle", "II. Tight bounding box", "III. Spotlight", 
                                                "IV. Segmentation", "V. Segmentation outline", "VI. Detection confidence", 
                                                "VII. Detection signal"))
str(likert$Design)
levels(likert$Design)

likert$Video <- as.factor(likert$Video)
likert$DetectMorePolyps <- ordered(likert$DetectMorePolyps, levels = c("Strongly disagree", "Disagree", "Somewhat disagree",
                                                                       "Neither agree nor disagree", "Somewhat agree",
                                                                       "Agree", "Strongly agree"))
likert$FasterPolyps <- ordered(likert$FasterPolyps, levels = c("Strongly disagree", "Disagree", "Somewhat disagree",
                                                                       "Neither agree nor disagree", "Somewhat agree",
                                                                       "Agree", "Strongly agree"))
likert$InterferePolypectomy <- ordered(likert$InterferePolypectomy, levels = c("Strongly disagree", "Disagree", "Somewhat disagree",
                                                                       "Neither agree nor disagree", "Somewhat agree",
                                                                       "Agree", "Strongly agree"))
likert$InterfereRegular <- ordered(likert$InterfereRegular, levels = c("Strongly disagree", "Disagree", "Somewhat disagree",
                                                                       "Neither agree nor disagree", "Somewhat agree",
                                                                       "Agree", "Strongly agree"))


# Question-wise sum and percentage calculation for each category
# Detect more
summarized_detectmore <- likert %>%
  group_by(Design, DetectMorePolyps) %>%
  dplyr::summarize(n = n()) %>%
  mutate(freq = n / sum(n))

dt2[, .(N = .N)]


summarized_detectmore$DetectMorePolyps <- as.factor(summarized_detectmore$DetectMorePolyps)
str(summarized_detectmore$DetectMorePolyps)
levels(summarized_detectmore$DetectMorePolyps)
levels(summarized_detectmore$Design)

# Locate faster
summarized_locate_faster <- likert %>%
  group_by(Design, FasterPolyps) %>%
  dplyr::summarize(n = n()) %>%
  mutate(freq = n / sum(n))

summarized_locate_faster$FasterPolyps <- as.factor(summarized_locate_faster$FasterPolyps)
levels(summarized_locate_faster$FasterPolyps)

# Interfere polypectomy
summarized_interfere_polypec <- likert %>%
  group_by(Design, InterferePolypectomy) %>%
  summarize(n = n()) %>%
  mutate(freq = n / sum(n))

summarized_interfere_polypec$InterferePolypectomy <- as.factor(summarized_interfere_polypec$InterferePolypectomy)

# Interfere regular display
summarized_interfere_regular <- likert %>%
  group_by(Design, InterfereRegular) %>%
  summarize(n = n()) %>%
  mutate(freq = n / sum(n))

summarized_interfere_regular$InterfereRegular <- as.factor(summarized_interfere_regular$InterfereRegular)



##### Basic descriptive analysis #####

# Gender
table(survey_data$Gender)

# Age - Removed for data anonymisation
#survey_data$Age <- as.numeric(survey_data$Age)
#mean(survey_data$Age)
#sd(survey_data$Age)

table(survey_data$How.likely.are.you.to.use.validated.Artificial.Intelligence.or.Computer.Aided.Diagnosis.software.to.help.detect.polyps.if.it.was.available.)
table(survey_data$How.would.you.describe.your.willingness.to.generally.incorporate.new.technology.into.endoscopy.)
table(survey_data$How.excited.are.you.about.the.development.of.Artificial.Intelligence.technology.for.endoscopy.)
table(survey_data$How.concerned.are.you.about.the.development.of.Artificial.Intelligence.technology.for.endoscopy.)
table(survey_data$How.likely.are.you.to.use.validated.Artificial.Intelligence.or.Computer.Aided.Diagnosis.software.to.help.detect.polyps.if.it.was.available.)

