likertColours <- c("#2166ac","#67a9cf","#d1e5f0","#f7f7f7","#fddbc7","#ef8a62","#b2182b")

levels(summarized_detectmore$DetectMorePolyps)
levels(summarized_detectmore$Design)
str(summarized_detectmore$Design)

str(likert)
likert$Design <- fct_rev(likert$Design)
levels(likert$Design)

# Detect more polyps
### ART anova
model = art(DetectMorePolyps ~ Design + Video + Design:Video, data = likert)
str(likert)
summary(model)
anova(model)

### post hoc
model.lm = artlm(model, "Design")
str(model.lm)
marginal = emmeans(model.lm, ~ Design)
pairs(marginal)
plot(marginal, comparisons = TRUE)
multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)

detectmoreVideo <- emmip(model.lm, Design ~ Video | Video, plotit = F)
detectmoreVideo$Question = "Detect more polyps"

b <- multcomp::cld(marginal, adjust = "bonferroni", Letters=letters, reversed = T)
b <- select(b, c(Design, .group, emmean))
b <- merge(summarized_detectmore, b, by=c("Design"), all=TRUE)
b$.group <- trimws(b$.group)

detect_plot <- ggplot(data = b) + 
  geom_bar(aes(x = reorder(Design, emmean), y = freq, fill = reorder(DetectMorePolyps, desc(DetectMorePolyps))),
                    position="stack", stat="identity") +
  geom_text(aes(label = .group, y = 1.015, x = Design), position = 'identity', hjust = 0, check_overlap = TRUE) + 
  coord_flip() +
  labs(title="I would detect more polyps using this marker", y="",x="") +
  scale_fill_manual(values = likertColours) +
  theme(plot.title = element_text(size=14, hjust=0.5),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 13, hjust=0),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        legend.title = element_blank(),
        panel.grid = element_blank(),
        panel.background = element_blank(),
        legend.text=element_text(size=13),
        legend.position = "bottom",
        legend.spacing.x = unit(0.1, 'cm')) +
  guides(fill = guide_legend(reverse = TRUE, nrow = 1))
detect_plot

b <- NULL

# Locate polyps faster
### ART anova
#####
model = art(FasterPolyps ~ Design + Video + Design:Video, data = likert)
summary(model)
anova(model)

### post hoc
model.lm = artlm(model, "Design")
marginal = emmeans(model.lm, ~ Design)
pairs(marginal)
plot(marginal, comparisons = TRUE)
multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)

detectfasterVideo <- emmip(model.lm, Design ~ Video | Video, plotit = FALSE)
detectfasterVideo$Question = "Detect polyps faster"
  
b <- multcomp::cld(marginal, adjust = "bonferroni", Letters=letters, reversed = T)
b <- select(b, c(Design, .group, emmean))
b <- merge(summarized_locate_faster, b, by=c("Design"), all=TRUE)
b$.group <- trimws(b$.group)

locate_plot <- ggplot(data = b) + 
  geom_bar(aes(x = reorder(Design, emmean), y = freq, fill = reorder(FasterPolyps, desc(FasterPolyps))), 
           position="stack", stat="identity") +
  geom_text(aes(label = .group, y = 1.015, x = Design), position = 'identity', hjust = 0, check_overlap = TRUE) + 
  coord_flip() +
  labs(title="I would locate polyps faster using this marker", y="",x="") +
  scale_fill_manual(values = likertColours) +
  theme(plot.title = element_text(size=14, hjust=0.5),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 13, hjust=0),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        legend.title = element_blank(),
        panel.grid = element_blank(),
        panel.background = element_blank(),
        legend.text=element_text(size=13),
        legend.position = "bottom",
        legend.spacing.x = unit(0.1, 'cm')) +
  guides(fill = guide_legend(reverse = TRUE, nrow = 1))
locate_plot

ggarrange(detect_plot,locate_plot,
                    ncol = 2, nrow = 1,
                    common.legend = T, legend = "bottom")
ggsave("plots/LikertMoreFaster.pdf", height = 10, width = 40, units = "cm")


# Flip colours around
likertColours <- c("#b2182b","#ef8a62","#fddbc7","#f7f7f7","#d1e5f0","#67a9cf","#2166ac")

# Interfere polypectomy
### ART anova
model = art(InterferePolypectomy ~ Design + Video + Design:Video, data = likert)
summary(model)
anova(model)

### post hoc
model.lm = artlm(model, "Design")
marginal = emmeans(model.lm, ~ Design)
pairs(marginal)
plot(marginal, comparisons = TRUE)
multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)

#emmip(model.lm, Design ~ Video | Video)
interferePolypVideo <- emmip(model.lm, Design ~ Video | Video, plotit = F)
interferePolypVideo$Question = "Interferes during polypectomy"

b <- multcomp::cld(marginal, adjust = "bonferroni", Letters=letters, reversed = F)
b <- select(b, c(Design, .group, emmean))
b <- merge(summarized_interfere_polypec, b, by=c("Design"), all=TRUE)
b$.group <- trimws(b$.group)

interfere_polypectomy_plot <- ggplot(data = b) + 
  geom_bar(aes(x = reorder(Design, desc(emmean)), y = freq, fill = InterferePolypectomy), 
                                   position="stack", stat="identity") +
  geom_text(aes(label = .group, y = 1.015, x = Design), position = 'identity', hjust = 0, check_overlap = TRUE) + 
  coord_flip() +
  labs(title="This visual marker would interfere during polypectomy", y="",x="") +
  scale_fill_manual(values = likertColours) +
  theme(plot.title = element_text(size=14, hjust=0.5),
        axis.text.x=element_blank(),
        axis.text.y = element_text(size = 13, hjust=0),
        axis.title.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        legend.title = element_blank(),
        panel.grid = element_blank(),
        panel.background = element_blank(),
        legend.text=element_text(size=13),
        legend.position = "bottom",
        legend.spacing.x = unit(0.1, 'cm')) +
  guides(fill = guide_legend(reverse = TRUE, nrow = 1))
interfere_polypectomy_plot

# Interfere visual display
### ART anova
model = art(InterfereRegular ~ Design + Video + Design:Video, data = likert)
summary(model)
anova(model)
### post hoc
model.lm = artlm(model, "Design")
marginal = emmeans(model.lm, ~ Design)
pairs(marginal)
plot(marginal, comparisons = TRUE)
multcomp::cld(marginal, adjust = "bonferroni", Letters=letters)

#emmip(model.lm, Design ~ Video | Video)
interfereGeneralVideo <- emmip(model.lm, Design ~ Video | Video, plotit = F)
interfereGeneralVideo$Question = "Interferes with regular display"

b <- multcomp::cld(marginal, adjust = "bonferroni", Letters=letters, reversed = F)
b <- select(b, c(Design, .group, emmean))
b <- merge(summarized_interfere_regular, b, by=c("Design"), all=TRUE)
b$.group <- trimws(b$.group)

interfere_regular_plot <- ggplot(data = b) + 
  geom_bar(aes(x = reorder(Design, desc(emmean)), y = freq, fill = InterfereRegular), 
                                        position="stack", stat="identity") +
  geom_text(aes(label = .group, y = 1.015, x = Design), position = 'identity', hjust = 0, check_overlap = TRUE) + 
  coord_flip() +
  labs(title="This visual marker would interfere with the regular visual display", y="",x="") +
  scale_fill_manual(values = likertColours) +
  theme(plot.title = element_text(size=14, hjust=0.5),
        axis.text.x=element_blank(),
        axis.text.y = element_text(size = 13, hjust=0),
        axis.title.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        legend.title = element_blank(),
        panel.grid = element_blank(),
        panel.background = element_blank(),
        legend.text=element_text(size=13),
        legend.position = "bottom",
        legend.spacing.x = unit(0.1, 'cm')) +
  guides(fill = guide_legend(reverse = TRUE, nrow = 1))
interfere_regular_plot

ggarrange(interfere_polypectomy_plot, interfere_regular_plot,
          ncol = 2, nrow = 1,
          common.legend = T, legend = "bottom")
ggsave("plots/LikertInterfere.pdf", height = 10, width = 40, units = "cm")


## Effect of video plot

EffectVideo <- merge(detectfasterVideo, detectmoreVideo, all = T)
EffectVideo <- merge(EffectVideo, interferePolypVideo, all = T)
EffectVideo <- merge(EffectVideo, interfereGeneralVideo, all = T)

str(EffectVideo)

levels(EffectVideo$Video)[levels(EffectVideo$Video)=="Easy"] <- "Apparent"
levels(EffectVideo$Video)[levels(EffectVideo$Video)=="Difficult"] <- "Challenging"
str(EffectVideo$Video)

levels(EffectVideo$Design)

levels(EffectVideo$Design) <- rev(levels(EffectVideo$Design))

ggplot(data = EffectVideo, aes(x = Video, y = yvar, 
                               group = Design, colour = Design, linetype = Design)) +
  geom_line() + 
  geom_point() +
  scale_x_discrete(limits = rev(levels(EffectVideo$Video))) +
  facet_grid(. ~ Question) +
  guides(colour = guide_legend(nrow = 2, ncol = 4, byrow = TRUE)) +
  ylab("Linear prediction") +
  theme(plot.title = element_text(size=13, hjust=0.5),
        #axis.text.y = element_text(size = 12, hjust=0),
        axis.title.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        legend.title = element_blank(),
        panel.grid.major.y = element_line(colour = "gray", size = 0.2, linetype = "dashed"),
        panel.grid = element_blank(),
        strip.background = element_blank(),
        panel.background = element_blank(),
        legend.position = "bottom",
        legend.spacing.x = unit(0.1, 'cm'),
        plot.margin = unit(c(-0.15,0.25,-0.3,0.25), "cm"),
        plot.background=element_rect(fill="transparent",colour=NA),
        legend.key = element_rect(fill = "transparent", colour = "transparent"))
ggsave("plots/EffectVideo.pdf", height = 6.5, width = 20, units = "cm")

