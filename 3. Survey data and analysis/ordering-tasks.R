data_summary <- function(x) {
  m <- mean(x)
  ymin <- m-sd(x)
  ymax <- m+sd(x)
  return(c(y=m,ymin=ymin,ymax=ymax))
}

### Ordering tasks
ranking_easy <- survey_data %>% select(Response.ID, ProfessionalRole, RankWideBoundingCircle, RankTightBoundingBox, 
                                       RankDetectionSignal, RankSpotlight, RankSegmentation, RankSegmentationOutline, RankDetectionConfidence) %>%
  melt(id.vars = c("Response.ID", "ProfessionalRole"))

ranking_easy$value <- as.numeric(ranking_easy$value)

levels(ranking_easy$variable)
ranking_easy$variable <- factor(ranking_easy$variable, labels=c("I. Wide bounding circle", "II. Tight bounding box", "VII. Detection signal", "III. Spotlight", "IV. Segmentation", "V. Segmentation outline", "VI. Detection confidence"))

a <- ranking_easy %>%
 group_by(variable) %>%
 summarise_all(funs(mean), na.rm = TRUE)
ranking_easy$variable <- factor(ranking_easy$variable, levels = a$variable[order(-a$value)])


ranking_easy$ProfessionalRole <- as.factor(ranking_easy$ProfessionalRole)
levels(ranking_easy$ProfessionalRole)
ranking_easy$ProfessionalRole <- fct_collapse(ranking_easy$ProfessionalRole,
                                              assistant = c("Assistant - Staff Nurse (not performing endoscopy)",
                                                            "Assistant - Healthcare Assistant (not performing endoscopy)"),
                                              endoscopist = c("Gastroenterology SpR", "Surgical SpR",
                                                              "Nurse Endoscopist", "Gastroenterology Consultant"))

str(ranking_easy$ProfessionalRole)
levels(ranking_easy$ProfessionalRole)[levels(ranking_easy$ProfessionalRole)=="assistant"] <- "Assistant"
levels(ranking_easy$ProfessionalRole)[levels(ranking_easy$ProfessionalRole)=="endoscopist"] <- "Endoscopist"

### ART anova
ranking_easy <- na.omit(ranking_easy)

model = art(value ~ variable + ProfessionalRole + variable:ProfessionalRole,
            data = ranking_easy)

#str(model)
summary(model)
anova(model)

# post hoc
model.lm = artlm(model, "variable")
marginal = emmeans(model.lm, ~ variable)

pairs(marginal)

plot(marginal, comparisons = TRUE)
multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)

b <- multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)
b <- select(b, c(variable, .group))
b <- merge(ranking_easy, b, by=c("variable"), all=TRUE)
b$.group <- trimws(b$.group)

survey_data %>% select("RankWideBoundingCircle", 
                         "RankTightBoundingBox", "RankSpotlight", 
                         "RankSegmentation", "RankSegmentationOutline",
                         "RankDetectionConfidence", "RankDetectionSignal") %>%
  summarise_all(funs(mean), na.rm = TRUE)

b %>%
  group_by(variable) %>%
  summarise(mean = mean(value))


str(b$value)
ggplot(b, aes(x = variable, y = value)) +
  xlab("") +
  scale_y_continuous(name = "Rank (higher is better)", breaks = c(1,2,3,4,5,6,7)) +
  geom_text(aes(label = .group, y = 8, x = variable), position = 'identity', hjust = 0, check_overlap = TRUE) + 
  coord_flip(ylim=c(1, 8)) +
  stat_summary(fun.y = "mean", 
               fun.ymin = function(x) {mean(x)-sd(x)/sqrt(length(x))}, 
               fun.ymax = function(x) {mean(x)+sd(x)/sqrt(length(x))},
               geom="pointrange", position = position_dodge(width = 0.35), fatten = 2, size = 1.8) +
  stat_summary(fun.y = "mean", 
               fun.ymin = function(x) {mean(x)-2*sd(x)/sqrt(length(x))}, 
               fun.ymax = function(x) {mean(x)+2*sd(x)/sqrt(length(x))},
               geom="pointrange", position = position_dodge(width = 0.35), fatten = 2, size = 1) +
  theme_classic() +
  theme(plot.title = element_text(size=13, hjust=0.5),
        axis.text = element_text(size = 12),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        axis.line = element_blank(),
        panel.grid.major.x = element_line(colour = "gray", size = 0.2, linetype = "dashed"),
        legend.position = "bottom",
        legend.spacing.x = unit(0.1, 'cm'))
ggsave("plots/MarkerRank.pdf", height = 7, width = 15, units = "cm")


ranking_easy %>%
   group_by(variable) %>%
   summarise(mean = mean(value), sd = sd(value))

a <- ranking_easy %>%
  spread(variable, value)

c <- matrix(as.matrix(a[,3:9]),
       nrow = nrow(a),
       dimnames = list(1 : nrow(a),
                       colnames(a[,3:9])))
boxplot(c)

### ART anova

ranking_easy <- na.omit(ranking_easy)

model = art(value ~ variable + ProfessionalRole + variable:ProfessionalRole,
            data = ranking_easy)

#str(model)
summary(model)
anova(model)

# post hoc
model.lm = artlm(model, "variable")

if(!require(emmeans)){install.packages("emmeans")}
library(emmeans)

marginal = emmeans(model.lm, ~ variable)

pairs(marginal)

plot(marginal, comparisons = TRUE)
#pwpp(marginal)
multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)


table_ranking <- table(ranking_easy$variable, ranking_easy$value)
table_ranking

ranking_easy$Response.ID <- as.factor(ranking_easy$Response.ID)
levels(ranking_easy$Response.ID)

# Colour
ranking_colour <- survey_data %>% select(Response.ID, ProfessionalRole, Black, Blue, Green, Orange, Purple, Red, Yellow) %>%
  melt(id.vars = c("Response.ID", "ProfessionalRole"))
ranking_colour$value <- as.numeric(ranking_colour$value)

# order by highest rank
a <- ranking_colour %>%
  group_by(variable) %>%
  summarise_all(funs(mean), na.rm = TRUE)
ranking_colour$variable <- factor(ranking_colour$variable, levels = a$variable[order(-a$value)])


ranking_colour$ProfessionalRole <- fct_collapse(ranking_colour$ProfessionalRole,
                                              assistant = c("Assistant - Staff Nurse (not performing endoscopy)",
                                                        "Assistant - Healthcare Assistant (not performing endoscopy)"),
                                              endoscopist = c("Gastroenterology SpR", "Surgical SpR",
                                                              "Nurse Endoscopist", "Gastroenterology Consultant"))

levels(ranking_colour$ProfessionalRole)[levels(ranking_colour$ProfessionalRole)=="assistant"] <- "Assistant"
levels(ranking_colour$ProfessionalRole)[levels(ranking_colour$ProfessionalRole)=="endoscopist"] <- "Endoscopist"

a <- ranking_colour %>%
  spread(variable, value)

c <- matrix(as.matrix(a[,3:9]),
            nrow = nrow(a),
            dimnames = list(1 : nrow(a),
                            colnames(a[,3:9])))
boxplot(c)

### ART anova
# Note, 4 participants did not complete the rank colour task
ranking_colour <- na.omit(ranking_colour)

model = art(value ~ variable + ProfessionalRole + variable:ProfessionalRole,
            data = ranking_colour)

summary(model)
anova(model)

# post hoc
model.lm = artlm(model, "variable")
marginal = emmeans(model.lm, ~ variable)

pairs(marginal)

plot(marginal, comparisons = TRUE)
multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)

b <- NULL
b <- multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)
b <- select(b, c(variable, .group))
b <- merge(ranking_colour, b, by=c("variable"), all=TRUE)
b$.group <- trimws(b$.group)

levels(b$variable)

ggplot(b, aes(x = variable, y = value)) +
  xlab("") +
  scale_x_discrete(limits=c("Blue", "Black", "Green", "Yellow", "Red", "Purple", "Orange")) +
  scale_y_continuous(name = "Rank (higher is better)", breaks = c(1,2,3,4,5,6,7)) +
  geom_text(aes(label = .group, y = 8, x = variable), position = 'identity', hjust = 0, check_overlap = TRUE) + 
  coord_flip(ylim=c(1, 8)) +
  stat_summary(fun.y = "mean", 
               fun.ymin = function(x) {mean(x)-sd(x)/sqrt(length(x))}, 
               fun.ymax = function(x) {mean(x)+sd(x)/sqrt(length(x))},
               geom="pointrange", position = position_dodge(width = 0.35), fatten = 2, size = 1.8) +
  stat_summary(fun.y = "mean", 
               fun.ymin = function(x) {mean(x)-2*sd(x)/sqrt(length(x))}, 
               fun.ymax = function(x) {mean(x)+2*sd(x)/sqrt(length(x))},
               geom="pointrange", position = position_dodge(width = 0.35), fatten = 2, size = 1) +
  theme_classic() +
  theme(plot.title = element_text(size=13, hjust=0.5),
        axis.text = element_text(size = 12),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        axis.line = element_blank(),
        panel.grid.major.x = element_line(colour = "gray", size = 0.2, linetype = "dashed"),
        legend.position = "bottom",
        legend.spacing.x = unit(0.1, 'cm'))
ggsave("plots/ColourRank.pdf", height = 7, width = 15, units = "cm")

