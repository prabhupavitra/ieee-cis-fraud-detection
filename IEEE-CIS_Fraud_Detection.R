##########################################################
# Create df set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(utils)) install.packages("utils", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(ipred)) install.packages("ipred", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(DMwR)) install.packages("DMwR", repos = "http://cran.us.r-project.org")
if(!require(vip)) install.packages("vip", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(plyr)
library(tidyverse)
library(caret)
library(data.table)
library(utils)
library(lubridate)
library(pROC)
library(ipred)
library(e1071)
library(xgboost)
library(reshape2)
library(xgboost)
library(DMwR)
library(vip)
library(randomForest)

# Download files from dropbox link
dl <- tempfile()
download.file("https://dl.dropboxusercontent.com/s/05kl7ogmpsx044j/ieee_cis_fraud_detection.zip", dl)

identification <- read_csv(unzip(dl, "ieee_cis_fraud_detection/train_identity.csv"))
transactions  <- read_csv(unzip(dl, "ieee_cis_fraud_detection/train_transaction.csv"))

fraud_det_set <- transactions %>% 
  left_join(identification,by="TransactionID")

# Validation set will be 10% of fraud_det_set data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = fraud_det_set$isFraud, times = 1, p = 0.1, list = FALSE)
df <- fraud_det_set[!row.names(fraud_det_set) %in% test_index,]
validation <- fraud_det_set[row.names(fraud_det_set) %in% test_index,]
validation_y <- select(validation,isFraud)
validation_x <- select(validation,-isFraud)

# Removing unwanted objects
rm(dl,identification, transactions,fraud_det_set,test_index,validation)


#################################################
##### Create Helper Functions
#################################################

# This function creates one-hot encoded columns of a selected column
one_hot_encoding <- function(df,colname){
  # Get the list of possible values category can have
  possible_val <- unique(df[[colname]])
  temp_df <- cbind(sapply(possible_val, function(g) {
    as.numeric(str_detect(df[[colname]], g))
  })
  )
  return(temp_df)
}

# This function performs label encoding of the column passed
label_encoding <- function(df,colname){
  levels <- sort(unique(df[[colname]]))
  temp_df <- as.integer(factor(df[[colname]], levels = levels))
  return (temp_df)
}

# This function replaces multiple values of a column with a single value
replace_columnval <- function(col,multiple,single){
  temp <- replace(col,str_detect(col,paste(multiple,collapse = "|")),single)
  return (temp)}

####################################################
# Data Visualizations for Exploratory Data Analysis
####################################################

### Response variable : isFraud

# Class Distribution of Fraudulent vs Legitimate Transactions 
ggplot(data = df, aes(x = factor(isFraud), fill = factor(isFraud))) +
  geom_bar(aes(y = (..count..)/sum(..count..)),position = "dodge",alpha=0.7) + 
  geom_text(aes(y = (..count..)/sum(..count..), 
                label = paste0(..count..," ","  (",round(prop.table(..count..) * 100,2), '%',")")), 
            stat = 'count', 
            position = position_dodge(.9), 
            size=2.5,vjust = -0.3) + 
  scale_fill_manual("",values=c("#66c2a5","#fc8d62"),labels=c("Legitimate","Fraud")) +
  labs(x = "", y = '') +
  scale_y_continuous(labels = scales::percent) +
  theme(legend.position = "bottom", 
        axis.text=element_blank(),
        axis.ticks = element_blank())

#########################################################################################################################
####### Quantitative Variables : TransactionAmt, dist1, dist2, TransactionDT, C1 ~ C14, D1 ~ D15 , Vxxx and id_01~id_11
#########################################################################################################################

# TransactionDT : timedelta from a given reference datetime (not an actual timestamp)
# Visualizing days from origin and associated total transaction amount
df %>% 
  dplyr::mutate(Day = round(df$TransactionDT/86400)) %>%
  select(isFraud,TransactionAmt,TransactionDT,Day) %>% 
  group_by(isFraud,Day) %>% 
  dplyr::summarise(txn_amt = sum(TransactionAmt)/1000) %>% 
  ggplot(aes(Day,txn_amt,color = factor(isFraud))) +
  geom_point() +
  geom_smooth() +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) +
  facet_grid(rows = vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  labs(x ="24-hour timedelta Relative from Origin",y="Total Transaction Amount (in Thousands)") +
  theme(legend.position = "none")


### Transaction Amount : transaction payment amount in USD
# Distribution of Transaction amount by fraudulent vs legitimate transactions
df %>% 
  ggplot(aes(TransactionAmt,color = factor(isFraud))) +
  geom_density() +
  facet_wrap(vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  scale_color_manual("", 
                     values=c("#66c2a5","#fc8d62")) +
  labs(y = "Density",x = "Transaction Amount (in USD)") +
  coord_cartesian(clip = "off") +
  theme(plot.title = element_text(hjust = 0.5,face="italic"),
        legend.position = "none")

# Normalized ecdf of transaction amount of Fraudulent transactions vs Legitimate transactions
df %>% 
  ggplot(aes(x = TransactionAmt,
             y =100*..y..,
             color=factor(isFraud))) +
  stat_ecdf(geom = "step", 
            size = 0.75, 
            pad = FALSE) +
  facet_wrap(vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  scale_x_continuous(breaks = seq(0,30000,5000)) +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) + 
  labs(x = "Transaction Amount (in USD)", y = "Cumulative Frequency") +
  coord_cartesian(clip = "off") +
  theme(legend.position = "none")


### dist1 and dist2 : Distance
# Visualizing dist1 and dist2 values  
df %>%
  select(dist1,dist2,isFraud) %>%
  gather("Dist","Values",-isFraud) %>% 
  ggplot(aes(Values,color = factor(isFraud))) +
  stat_ecdf(geom = "step", 
            size = 0.75, 
            pad = FALSE) +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) +
  facet_grid(isFraud~Dist, 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  labs(x="Distance",y = "Cumulative Frequency") +
  theme(legend.position = "none")


### C1~C14 variables dataframe : counting, such as how many addresses are found to be associated with the payment card,etc.
c_vars_df <-select(df,names(df)[str_detect(names(df),"^C[0-9]+")])

# Pairwise Correlation heatmap of C1~C14
cor(c_vars_df,
    use="pairwise.complete.obs") %>%
  melt() %>%
  dplyr::mutate(Var1=as.numeric(gsub("C([0-9]+)","\\1",Var1)),
                Var2=as.numeric(gsub("C([0-9]+)","\\1",Var2))) %>% 
  ggplot(aes(factor(Var1),factor(Var2),fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#d7301f",mid="white", high = "#045a8d") +
  labs(x="",y="") +
  scale_x_discrete(labels =function(x) paste0("C",x)) +
  scale_y_discrete(labels =function(x) paste0("C",x))

# Redundant features among C1~C14
c_vars_drop <- names(c_vars_df)[!str_detect(names(c_vars_df),"C([1,3,5,13]$)")]

# Remove unwanted variables
rm(c_vars_df)



### D1 ~ D15 : timedelta, such as days between previous transaction, etc.
# D1 ~ D15 variables dataframe
d_vars_df <- select(df,names(df)[str_detect(names(df),"^D[0-9]+")])

# Pairwise Correlation heatmap of D1~D15
cor(d_vars_df,
    use="pairwise.complete.obs") %>%
  melt() %>%
  dplyr::mutate(Var1=as.numeric(gsub("D([0-9]+)","\\1",Var1)),
                Var2=as.numeric(gsub("D([0-9]+)","\\1",Var2))) %>% 
  ggplot(aes(factor(Var1),factor(Var2),fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#d7301f",mid="white", high = "#045a8d") +
  labs(x="",y="") +
  scale_x_discrete(labels =function(x) paste0("D",x)) +
  scale_y_discrete(labels =function(x) paste0("D",x))

# Remove variables with more than 85% values missing 
d_vars_missingvalues <-
  data.frame(Var = names(colMeans(is.na(df))),
             missingPercent = colMeans(is.na(df))) %>% 
  filter(grepl("^D([0-9]+)",Var)) %>% 
  filter(!round(missingPercent,2) < 0.85) %>% 
  pull(Var)

# Remove unwanted variables
rm(d_vars_df)



### Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.
# Vxxx variables dataframe
vxxx_vars_df <- select(df,names(df)[str_detect(names(df),"^V")])

# Visualizing pairwise correlation of Vxxx features using heatmap
cor(vxxx_vars_df,
    use="pairwise.complete.obs") %>%
  melt() %>%
  dplyr::mutate(Var1=as.numeric(gsub("V([0-9]+)","\\1",Var1)),
                Var2=as.numeric(gsub("V([0-9]+)","\\1",Var2))) %>%
  ggplot(aes(Var1,Var2,fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#d7301f",mid="white", high = "#045a8d") +
  labs(x="",y="") +
  scale_x_continuous(breaks=seq(1,339,10),labels =function(x) paste0("V",x)) +
  scale_y_continuous(breaks=seq(1,339,10),labels =function(x) paste0("V",x)) +
  theme(axis.text.x = element_text(angle = 90))

# Remove variables with more than 70% values missing 
vxxx_vars_nomissingvalues <-
  data.frame(Var = names(colMeans(is.na(df))),
             missingPercent = colMeans(is.na(df))) %>% 
  filter(grepl("^V([0-9]+)",Var)) %>% 
  filter(!round(missingPercent,2) >= 0.7) %>% 
  pull(Var)

# Principal component analysis of Vxxx features
pca_Vxxx <- prcomp(na.omit(select(df,all_of(vxxx_vars_nomissingvalues))), center = FALSE, scale = FALSE)

# First 23 components explain much of the variability
#summary(pca_Vxxx)$importance[,1:16]

# Features dropped among Vxxx features
vxxx_vars_drop <- names(vxxx_vars_df)[!names(vxxx_vars_df) %in% vxxx_vars_nomissingvalues[1:16]]

# Remove unwanted variables
rm(vxxx_vars_df,pca_Vxxx,vxxx_vars_nomissingvalues)



### id_01 ~ id11
# Variables with more than 99% of the values as NA
missing_vars <- data.frame(Var = names(colMeans(is.na(df))),
                           missingPercent = colMeans(is.na(df))) %>% 
  filter(round(missingPercent,2) >= 0.99) %>%
  pull(Var)

# id_01 ~ id_38 variables dataframevariables dataframe
id_vars_df <- select(df,names(df)[str_detect(names(df),"^id_[0-9]+")])

# Remove variables with more than 99% values missing
numeric_id_vars_df <- select(df,names(select(df,names(id_vars_df)[1:11],-all_of(missing_vars))))

# Pairwise Correlation heatmap of id_01~id11
cor(numeric_id_vars_df,
    use="pairwise.complete.obs") %>%
  melt() %>%
  dplyr::mutate(Var1=as.numeric(gsub("id_([0-9]+)","\\1",Var1)),
                Var2=as.numeric(gsub("id_([0-9]+)","\\1",Var2))) %>% 
  ggplot(aes(factor(Var1),factor(Var2),fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#d7301f",mid="white", high = "#045a8d") +
  labs(x="",y="") +
  scale_x_discrete(labels =function(x) paste0("id_",x)) +
  scale_y_discrete(labels =function(x) paste0("id_",x))

rm(numeric_id_vars_df,id_vars_df)



#########################################################################################################################################################
####### Categorical Features : ProductCD, card1 ~ card6, M1~M9, addr1 ~ addr2, P_emaildomain and R_emaildomain, Device Type, Device Info and id12 ~ id38
#########################################################################################################################################################

### ProductCD : product code, the product for each transaction  
# Fraudulent transactions vs legitimate transactions grouped by ProductCD
df %>% 
  group_by(ProductCD,isFraud) %>%
  dplyr::summarise(num_txn = n()) %>%
  data.frame() %>% 
  dplyr::mutate(prop = num_txn/sum(num_txn)) %>% 
  ggplot(aes(x = ProductCD,y = prop, fill = factor(isFraud))) + 
  geom_bar(stat="identity",
           position=position_dodge(),
           alpha=0.7,
           color="white") +
  geom_text(aes(x = ProductCD, y = prop,
                label = paste0(round(prop * 100,2), '%')),
            position=position_dodge(0.9), 
            size=2.5,vjust = -1) +
  labs(x = "Product Code",y = "% of Number of Transactions") +
  scale_y_continuous(labels = scales::percent,limits = c(0,1)) +
  scale_fill_manual("",values=c("#66c2a5","#fc8d62"),labels=c("Legitimate","Fraud")) +
  theme(legend.position = "bottom")


# Total Transaction amount of Fraudulent transactions vs legitimate transactions grouped by ProductCD
df %>% 
  group_by(isFraud,ProductCD) %>%
  dplyr::summarise(Txn_amount = sum(TransactionAmt)/1000) %>% 
  ggplot(aes(x = ProductCD,y = Txn_amount, fill = factor(isFraud))) + 
  geom_bar(stat="identity",
           position=position_dodge(),
           alpha=0.7,
           color="white") +
  geom_text(aes(x = ProductCD, y = Txn_amount,
                label = paste0(round(Txn_amount,2),"$")),
            position=position_dodge(0.9), 
            size = 2,vjust=-0.1) +
  facet_wrap(ProductCD ~.,scales = "free") +
  scale_fill_manual("",values=c("#66c2a5","#fc8d62"),labels =c("Legitimate","Fraud")) +
  labs(y="Total Transaction Amount (in Thousands)",x = "") +
  theme(legend.position = "bottom")

# Distribution of Transaction amount of Fraudulent transactions vs legitimate transactions grouped by ProductCD
df %>% 
  ggplot(aes(x = ProductCD,y = TransactionAmt, fill = factor(isFraud))) + 
  geom_boxplot(outlier.colour="red",
               outlier.shape=1,
               alpha=0.7) +
  labs(y="Transaction Amount") +
  ylim(0,1500) +
  coord_flip() +
  scale_fill_manual("",
                    values=c("#66c2a5","#fc8d62"),
                    labels = c("Legitimate", "Fraud")) +
  theme(legend.position = "bottom")




### card1 - card6 : payment card information, such as card type, card category, issue bank, country, etc.
# Visualizing  variables Card 1, Card2 and Card3 and Card5 and associated number of transactions
df %>% 
  select(isFraud,card1,card2,card3,card5) %>%
  gather("Card","Values",-isFraud) %>% 
  group_by(Card,isFraud,Values) %>%
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(x = Values,y = num_txn,color=factor(Card))) +
  geom_point(alpha=0.7) +
  scale_color_manual("",values=c("#66c2a5","#fc8d62","#8da0cb","#ffd92f","#bc80bd")) +
  facet_wrap(Card ~ isFraud, 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud")),
             ncol=2) +
  labs(x ="card1 values",y="Number of Transactions") +
  theme(legend.position = "none")

# Visualizing  card1, card2, card3 and card5 values and associated transaction amount
df %>% 
  select(isFraud,TransactionAmt,card1,card2,card3,card5) %>%
  gather("Card","Values",-isFraud,-TransactionAmt) %>% 
  group_by(Card,isFraud,Values) %>%
  dplyr::summarise(txn_amt = sum(TransactionAmt)/1000) %>%
  ggplot(aes(x = Values,y = txn_amt,color=factor(Card))) +
  geom_point(alpha=0.7) +
  scale_color_manual("",values=c("#66c2a5","#fc8d62","#8da0cb","#ffd92f","#bc80bd")) +
  facet_wrap(Card ~ isFraud, 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud")),
             ncol=2) +
  labs(x ="card1 values",y="Number of Transactions") +
  theme(legend.position = "none")


# Visualizing variables card4 and card6 
# Number of transactions for each Card type and network, grouped by transaction type(Fraudulent/legitimate)
df %>%
  dplyr::mutate(card4 = replace_columnval(card4,
                                          unique(df$card4)[!str_detect(unique(df$card4),
                                                                            "visa|master|discover|amer")],"Others"),
                card6 = replace_columnval(card6,unique(df$card6)[!str_detect(unique(df$card6),
                                    "debit|credit")],"Others")) %>%
  group_by(isFraud,card4,card6) %>%
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(y = num_txn,x = card6, fill=factor(card6))) +
  geom_bar(stat="identity",alpha=0.8,position=position_dodge()) +
  facet_wrap(isFraud ~ card4 , 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"),
                                 card4 = c("Other Networks"= "Other \n Networks","american express"="Amex","discover"="Discover","mastercard"="MasterCard","visa"="Visa")),
             ncol = 5) +
  scale_fill_manual("",values=c("#66c2a5","#fc8d62","#8da0cb","#ffd92f","#bc80bd")) +
  labs(y="Number of Transactions",x = "Card Type") +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 90))

# Total transaction amount for each Card type and network, grouped by transaction type(Fraudulent/legitimate)
df %>% 
  dplyr::mutate(card4 = replace_columnval(card4,
                                          unique(df$card4)[!str_detect(unique(df$card4),
                                                                       "visa|master|discover|amer")],"Others"),
                card6 = replace_columnval(card6,unique(df$card6)[!str_detect(unique(df$card6),
                                                                             "debit|credit")],"Others")) %>% 
  group_by(isFraud,card4,card6) %>%
  dplyr::summarise(Txn_amount = sum(TransactionAmt)/1000) %>%
  ggplot(aes(y = Txn_amount,x = card6, fill=factor(card6))) +
  geom_bar(stat="identity",alpha=0.8) +#position=position_dodge()
  facet_wrap(isFraud ~ card4 , 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"),
                                 card4 = c("Other Networks"= "Other Networks","american express"="Amex","discover"="Discover","mastercard"="MasterCard","visa"="Visa")),
             ncol = 5) +
  labs(y="Transaction Amount (in Thousands)",x = "Card Type") +
  scale_fill_manual("",values=c("#66c2a5","#fc8d62","#8da0cb","#ffd92f","#bc80bd")) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 90))

### M1 ~ M9 (logical) : match, such as names on card and address, etc.
# Visualizing M1~ M9 values and associated number of transactions
df %>% 
  select(isFraud,M1:M3,M5:M9) %>%
  gather("Var","values",-isFraud) %>%
  group_by(Var,isFraud,values) %>%
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(x = Var,y = num_txn,fill=factor(Var))) +
  geom_bar(stat="identity",
           alpha=0.65,
           position = position_dodge()) +
  scale_fill_brewer(type="qual",palette=3) +
  facet_wrap(values ~ isFraud, 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud")),
             ncol=2) +
  labs(x ="",y="Number of Transactions") +
  theme(legend.position = "bottom",
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

# Visualizing M1~M9 values and associated total transaction amount
df %>% 
  select(TransactionAmt,isFraud,M1:M3,M5:M9) %>%
  gather("Var","values",-TransactionAmt,-isFraud) %>%
  group_by(Var,isFraud,values) %>%
  dplyr::summarise(txn_amt = sum(TransactionAmt)/1000) %>%
  ggplot(aes(Var,y=txn_amt,fill=factor(Var))) +
  scale_fill_brewer(type="qual",palette=3) +
  geom_bar(stat="identity",
           alpha=0.65,
           position = position_dodge()) +
  facet_wrap(values ~ isFraud, 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud")),
             ncol=2) +
  labs(x="",y="Total Transaction Amount (in Thousands)") +
  theme(legend.position = "bottom")


### M4 (Character variable) 
# Visualizing M4 values and associated number of transactions 
df %>% 
  select(isFraud,M4) %>%
  dplyr::mutate(M4 = replace(M4,is.na(M4),"(Blank)")) %>% 
  group_by(isFraud,M4) %>%
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(x = M4,y = num_txn,color = factor(isFraud),fill=factor(M4))) +
  geom_bar(stat="identity",
           position = position_dodge()) +
  scale_fill_brewer(type="qual",palette=1) +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) +
  facet_wrap(vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud")),
             ncol=2) +
  labs(x ="M4 values",y="Number of Transactions") +
  theme(legend.position = "none")

# Visualizing M4 values and associated total transaction amount
df %>% 
  select(TransactionAmt,isFraud,M4) %>%
  dplyr::mutate(M4 = replace(M4,is.na(M4),"(Blank)")) %>% 
  group_by(isFraud,M4) %>%
  dplyr::summarise(txn_amt = sum(TransactionAmt)/1000) %>%
  ggplot(aes(x = M4,y = txn_amt,color = factor(isFraud),fill=factor(M4))) +
  geom_bar(stat="identity",
           position = position_dodge()) +
  scale_fill_brewer(type="qual",palette=1) +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) +
  facet_wrap(vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud")),
             ncol=2) +
  labs(x ="M4 values",y="Total Transaction Amount (in Thousands)") +
  theme(legend.position = "none")



### addr1 and addr2 : Address
# Visualizing addr1 values and associated number of transactions
df %>%
  group_by(isFraud,addr1) %>%
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(addr1,num_txn,color=factor(isFraud))) +
  geom_point() +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) +
  facet_grid(rows = vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  labs(x ="addr1 values",y="Number of Transactions") +
  theme(legend.position = "none")

# Visualizing addr1 values and associated total transaction amount
df %>%
  group_by(isFraud,TransactionAmt,addr1) %>%
  dplyr::summarise(txn_amt = sum(TransactionAmt)) %>%
  ggplot(aes(addr1,txn_amt,color=factor(isFraud))) +
  geom_point() +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) +
  facet_grid(rows = vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  labs(x ="addr1 values",y="Total Transaction Amount (in Thousands)") +
  theme(legend.position = "none")


# Visualizing addr2 values and associated number of transactions
df %>%
  group_by(isFraud,addr2) %>%
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(addr2,num_txn,color=factor(isFraud))) +
  geom_point() +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) +
  geom_vline(xintercept=87,linetype = "longdash",color="grey") +
  geom_text(aes(87,y=-500,label=87,hjust=1.5),
            size=2.5,family="Courier", fontface="italic",color="darkgrey") +
  facet_grid(rows = vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  labs(x ="addr2 values",y="Number of Transactions") +
  theme(legend.position = "none")

# Visualizing addr2 values and associated total transaction amount
df %>%
  group_by(isFraud,TransactionAmt,addr2) %>%
  dplyr::summarise(txn_amt = sum(TransactionAmt)) %>%
  ggplot(aes(addr2,txn_amt,color=factor(isFraud))) +
  geom_point() +
  scale_color_manual("",values=c("#66c2a5","#fc8d62")) +
  geom_vline(xintercept=87,linetype = "longdash",color="grey") +
  geom_text(aes(87,y=-500,label=87,hjust=1.5),
            size=2.5,family="Courier", fontface="italic",color="darkgrey") +
  facet_grid(rows = vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  labs(x ="addr2 values",y="Total Transaction Amount (in Thousands)") +
  theme(legend.position = "none")


### P_emaildomain: purchaser email domain
# Visualizing purchaser email domain of approximately 99% of the transactions.
df %>%
  dplyr::mutate(P_emaildomain = replace_columnval(P_emaildomain,c("live","outlook","hotmail","msn"),"Microsoft"),
                # Replaces 8 Yahoo domains 
                P_emaildomain = replace_columnval(P_emaildomain,c("yahoo","ymail"),"Yahoo"),
                # Replaces 3 icloud domains 
                P_emaildomain = replace_columnval(P_emaildomain,c("me.com","mac.com","icloud.com"),"iCloud"),
                # Replaces 2 AOL domains 
                P_emaildomain = replace_columnval(P_emaildomain,c("aol","aim"),"AOL"),
                P_emaildomain = replace_columnval(P_emaildomain,c("gmail.com","gmail"),"Gmail"),
                P_emaildomain = replace_columnval(P_emaildomain,c(".net"),"NetworkingDomains"),
                P_emaildomain = replace_columnval(P_emaildomain,unique(df$P_emaildomain)[!str_detect(unique(df$P_emaildomain),
                                                                                                     "Gmail|Yahoo|Networking|Microsoft|AOL|iCloud")]
                                                  ,"Others")) %>%
  dplyr::mutate(P_emaildomain = replace(P_emaildomain,is.na(P_emaildomain),"Unknown")) %>%
  group_by(isFraud,P_emaildomain) %>%
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(reorder(P_emaildomain,num_txn),num_txn,fill=factor(isFraud))) +
  geom_bar(stat="identity",position=position_dodge()) +
  scale_fill_manual("",values=c("#66c2a5","#fc8d62")) +
  facet_wrap(vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  coord_flip() +
  labs(x ="Purchaser email domain",y="Number of transactions") +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 90))



### R_emaildomain : recipient email domain
# Visualizing recipient email domain of approximately 99% of the transactions.
df %>%
  dplyr::mutate(R_emaildomain = replace_columnval(R_emaildomain,c("live","outlook","hotmail","msn"),"Microsoft"),
                R_emaildomain = replace_columnval(R_emaildomain,c("yahoo","ymail"),"Yahoo"),
                R_emaildomain = replace_columnval(R_emaildomain,c("me.com","mac.com","icloud.com"),"iCloud"),
                R_emaildomain = replace_columnval(R_emaildomain,c("aol","aim"),"AOL"),
                R_emaildomain = replace_columnval(R_emaildomain,c("gmail.com","gmail"),"Gmail"),
                R_emaildomain = replace_columnval(R_emaildomain,c(".net"),"NetworkingDomains"),
                R_emaildomain = 
                  replace_columnval(R_emaildomain,
                                    unique(df$R_emaildomain)[!str_detect(unique(df$R_emaildomain),
                                                                         "Gmail|Yahoo|Networking|Microsoft|AOL|iCloud")],"Others")) %>%
  dplyr::mutate(R_emaildomain = replace(R_emaildomain,is.na(R_emaildomain),"Unknown")) %>%
  group_by(isFraud,R_emaildomain) %>%
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(reorder(R_emaildomain,num_txn),num_txn,fill=factor(isFraud))) +
  geom_bar(stat="identity",position=position_dodge()) +
  scale_fill_manual("",values=c("#66c2a5","#fc8d62")) +
  facet_wrap(vars(isFraud),
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  coord_flip() +
  labs(x ="Recepient email domain",y="Number of transactions") +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 90))


### Device Type : type of device used for transaction (mobile/desktop/Unknown)
df %>%
  dplyr::mutate(DeviceType = replace(DeviceType,is.na(DeviceType),"Unknown")) %>%
  group_by(isFraud,DeviceType) %>% 
  dplyr::summarise(num_txn = n()) %>%
  ggplot(aes(y = num_txn,x = DeviceType, fill=factor(isFraud))) +
  geom_bar(stat="identity",alpha=0.8,position=position_dodge()) +
  facet_wrap(vars(isFraud), 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"))) +
  scale_fill_manual("",values=c("#66c2a5","#fc8d62")) +
  labs(y="Number of Transactions",x = "Device Type") +
  theme(legend.position = "none")


### Device Info 
# Visualizing Device information of approximately 99% of dataset (Number of transactions > 100)
df %>%
  select(DeviceType,DeviceInfo,isFraud) %>%
  dplyr::mutate(DeviceInfo = replace_columnval(DeviceInfo,c("SM","SAMSUNG","samsung"),"Samsung"),
                DeviceInfo = replace_columnval(DeviceInfo,c("HTC"),"HTC"),
                DeviceInfo = replace_columnval(DeviceInfo,c("Moto","Motorola","MOTOROLA","XT1635"),"Motorola"),
                DeviceInfo = replace_columnval(DeviceInfo,c("HUAWEI","Huawei","hi6210"),"Huawei"),
                DeviceInfo = replace_columnval(DeviceInfo,c("LG"),"LG"),
                DeviceInfo = replace(DeviceInfo,is.na(DeviceInfo),"Unknown")) %>%
  dplyr::mutate(DeviceType = replace(DeviceType,is.na(DeviceType),"Unknown")) %>% 
  group_by(DeviceType,DeviceInfo,isFraud) %>%
  dplyr::summarise(num=n()) %>%
  arrange(-num) %>% 
  filter(num>100) %>%  
  ggplot(aes(y = num,x = DeviceInfo, fill=factor(isFraud))) +
  geom_bar(stat="identity",alpha=0.8,position=position_dodge()) +
  facet_wrap(isFraud~DeviceType, 
             scales = "free",
             labeller = labeller(isFraud = c("0"= "Legitimate","1" = "Fraud"),
                                 DeviceType = c("desktop"="Desktop","mobile"="Mobile","Unknown"="Unknown"))) +
  scale_fill_manual("",values=c("#66c2a5","#fc8d62","#8da0cb")) +
  labs(y="Number of Transactions",x = "Device Information") +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 90))

#################################################
##### Feature Selection & Imputing missing data
#################################################
# 434 Features in original dataset 

##### Initial Feature Selection by EDA
eliminated_features <- c(c_vars_drop,d_vars_missingvalues,vxxx_vars_drop,missing_vars)
rm(c_vars_drop,d_vars_missingvalues,vxxx_vars_drop,missing_vars)

##### Feature Engineering : Label Encoding, One-hot encoding, numeric encoding

# This function encodes the dataset and imputes necessary missing values
feature_engg_fraud_det <- function(temp){
  # Eliminated 350 variables after EDA; Remaining 84 variables
  numeric_df <- select(temp,-all_of(eliminated_features)) %>% select_if(is.numeric)  #  Numeric vectors
  char_df <- select(temp,-all_of(eliminated_features)) %>% select_if(is.character)   #  Character vectors
  logical_df <- select(temp,-all_of(eliminated_features)) %>% select_if(is.logical) # 12 Logical vectors
  
# Merging email domains, card values and imputing missing categorical values
char_df <- char_df %>%
              dplyr::mutate(card6 = replace_columnval(card6,
                                                      unique(char_df$card6)[!str_detect(unique(char_df$card6),
                                                                            "debit|credit")],"Others")) %>%
              dplyr::mutate(card4 = replace_columnval(card4,
                                                      unique(char_df$card4)[!str_detect(unique(char_df$card4),
                                                                                        "visa|master|discover|amer")],
                            "Others")) %>%
              dplyr::mutate(P_emaildomain = replace_columnval(P_emaildomain,
                                                              c("live","outlook","hotmail","msn"),
                                                              "Microsoft"),
                           # Replaces Yahoo domains 
                            P_emaildomain = replace_columnval(P_emaildomain,
                                                  c("yahoo","ymail"),
                                                  "Yahoo"),
                           # Replaces icloud domains 
                           P_emaildomain = replace_columnval(P_emaildomain,c("me.com","mac.com","icloud.com"),"iCloud"),
                           # Replaces AOL domains 
                           P_emaildomain = replace_columnval(P_emaildomain,
                                                             c("aol","aim"),
                                                             "AOL"),
                           P_emaildomain = replace_columnval(P_emaildomain,
                                                             c("gmail.com","gmail"),
                                                             "Gmail"),
                           P_emaildomain = replace_columnval(P_emaildomain,
                                                             c(".net"),
                                                             "NetworkingDomains"),
                           P_emaildomain = replace_columnval(P_emaildomain,
                                                             unique(char_df$P_emaildomain)[!str_detect(unique(char_df$P_emaildomain),
                                                                                                       "Gmail|Yahoo|Networking|Microsoft|AOL|iCloud")]
                                                             ,"Others")) %>%
             dplyr::mutate(R_emaildomain = replace_columnval(R_emaildomain,
                                                             c("live","outlook","hotmail","msn"),
                                                             "Microsoft"),
                R_emaildomain = replace_columnval(R_emaildomain,
                                                  c("yahoo","ymail"),
                                                  "Yahoo"),
                R_emaildomain = replace_columnval(R_emaildomain,
                                                  c("me.com","mac.com","icloud.com"),
                                                  "iCloud"),
                R_emaildomain = replace_columnval(R_emaildomain,
                                                  c("aol","aim"),
                                                  "AOL"),
                R_emaildomain = replace_columnval(R_emaildomain,
                                                  c("gmail.com","gmail"),
                                                  "Gmail"),
                R_emaildomain = replace_columnval(R_emaildomain,
                                                  c(".net"),
                                                  "NetworkingDomains"),
                R_emaildomain = replace_columnval(R_emaildomain,
                                                  unique(char_df$R_emaildomain)[!str_detect(unique(char_df$R_emaildomain),
                                                                                                          "Gmail|Yahoo|Networking|Microsoft|AOL|iCloud")]
                                                  ,"Others")) %>%
             dplyr::mutate(DeviceType =  replace(DeviceType,is.na(DeviceType),"Unknown"),
                           DeviceInfo =  replace(DeviceInfo,is.na(DeviceInfo),"Unknown"),
                           card4 = replace(card4,is.na(card4),"Unknown"),
                           card6 = replace(card6,is.na(card6),"Unknown"),
                           P_emaildomain = replace(P_emaildomain,is.na(P_emaildomain),"Unknown"),
                           R_emaildomain = replace(R_emaildomain,is.na(R_emaildomain),"Unknown"),
                           M4 =  replace(M4,is.na(M4),"Unknown"),
                           id_12 = replace(id_12,is.na(id_12),"Unknown"),
                           id_15 = replace(id_15,is.na(id_15),"Unknown"),
                           id_16 = replace(id_16,is.na(id_16),"Unknown"),
                           id_28 = replace(id_28,is.na(id_28),"Unknown"),
                           id_29 = replace(id_29,is.na(id_29),"Unknown"),
                           id_30 = replace(id_30,is.na(id_30),"Unknown"),
                           id_31 = replace(id_31,is.na(id_31),"Unknown"),
                           id_34 = replace(id_34,is.na(id_34),"Unknown")) %>%
  # Dropping resolution column
  select(-id_33)


  # Binding columns of the dataset :  106 variables
  temp_df <-
    # 5 character vectors encoded as 28 one-hot encoded variables
    cbind(as.data.frame(sapply(names(char_df)[str_detect(names(char_df),
                                                         "emaildomain|card|Product")],
                               function(x) one_hot_encoding(char_df,x))),
          # 11 character variables encoded with label encoding to 11 variables
          as.data.frame(sapply(names(char_df)[!str_detect(names(char_df),
                                                          "emaildomain|card|Product")],
                               function(x) label_encoding(char_df,x))),
          # 12 logical vectors encoded as 12 numeric encoding of logical variables
          as.data.frame(sapply(names(logical_df), 
                               function(x) as.numeric(logical_df[[x]]))),
          # 55 numeric variables 
          numeric_df)
  
  # Imputing numerical NA as -999
  temp_df <- temp_df %>% dplyr::mutate_if(is.numeric, list(~replace_na(., -999)))
  
  return (temp_df)
}

##########################################################
# Create train set, test set (Imbalanced)
##########################################################
## Create train and test sets for cross validation
df <- feature_engg_fraud_det(df) 
set.seed(1234, sample.kind="Rounding") 
# Create train set and test set
test_index <- createDataPartition(y = df$isFraud, times = 1 ,p=0.1, list = FALSE)

imbal_train_set <- df[!row.names(df) %in% test_index,]
imbal_test_set <-  df[row.names(df) %in% test_index,]

# Remove unwanted variables
rm(test_index)

################################################################
# Model Selection 
################################################################

# Computational control parameters for caret::train function
fitControl = trainControl(method = "repeatedcv", # Repeated Cross validation 
                          number = 5,  #  controls the number of folds in K-fold cross-validation
                       #   repeats = 5, # Repeats K-fold cross-validation
                          verboseIter = FALSE, # A logical for printing a training log.
                          returnData = FALSE, #  A logical for saving the data into a slot called trainingData
                          allowParallel = TRUE, # Use a parallel processing if available
                          sampling = "down", # Balancing imbalanced dataset using random downsampling
                          classProbs = TRUE, #  class probabilities be computed for classification models
                          summaryFunction = twoClassSummary) # a function to computed alternate performance summaries.

#### Model 1 : Bagged CART

# Setting seed for reproducibility
set.seed(231, sample.kind="Rounding") 

# Train the model to find the best tune for bagged CART; 
# Used default nbagg=25 for faster computation
fit.bagcart <-
  train(
    make.names(isFraud)~.,
    data = imbal_train_set,  
    trControl = fitControl,
    method = "treebag",
    metric ="ROC",
    tree_method = "hist",
    nthread = 2,
    verbose = TRUE)

# Predictions
pred.bagcart <- predict(fit.bagcart,
                           imbal_test_set %>% select(-isFraud),
                           type="prob")  

# ROC for Bagged CART
roctreebag_down <-  roc(imbal_test_set$isFraud,
                        factor(pred.bagcart$X1, ordered = TRUE),
                        plot=TRUE,
                        print.auc=TRUE,
                        col = "#08519c",
                        legacy.axes=TRUE,
                        main="ROC Curve for Bagged CART",
                        levels = c("0", "1"))

roctreebag_down 

# Displaying Results of bagged CART
auc_results <- tibble(method = "Bagged CART", AUC = roctreebag_down$auc)
auc_results %>% knitr::kable()

# Variable importance plot for bagged CART
vip::vip(fit.bagcart, num_features = 40, geom = "point")

# Model 2 : Random Forest

# mtry (The number of features to consider at any given split) for Classification models 
mtry = floor(sqrt(ncol(imbal_train_set)))

# Creating hyperparameter grid search for Random Forest
rfgrid <- expand.grid(.mtry=mtry)

# Setting seed for reproducibility 
set.seed(112, sample.kind="Rounding") 

# Train the model to find the best tune for Random Forest
fit.randomforest = train(
  make.names(isFraud)~.,
  data = imbal_train_set,  
  trControl = fitControl,
  tuneGrid = rfgrid,
  method = "rf",
  tree_method = "hist",
  metric ="ROC",
  verbose = TRUE)

# Predict with imbalanced test set
pred.rf <- predict(fit.randomforest,
                        imbal_test_set %>% select(-isFraud),
                        type = "prob")

# Generating ROC for the Random Forest model
rf_down <- roc(imbal_test_set$isFraud,
                factor(pred.rf$X1,ordered=TRUE),
                plot=TRUE,
                print.auc=TRUE,
                col = "#6a51a3",  
                legacy.axes=TRUE,
                levels = c("0", "1"))

rf_down

# Displaying Results of Random Forest
auc_results <- rbind(auc_results,tibble(method = "Random Forest", AUC =rf_down$auc))
auc_results %>% knitr::kable()

# Variable importance plot for Random Forest
vip::vip(fit.randomforest, num_features = 40, geom = "point")

# Model 3 : XGBOOST (eXtreme Gradient Boosting)

# Creating hyperparameter grid search for XGBOOST Hyperparameters with faster learning rate of 0.1
xgbGrid = expand.grid(nrounds = 500,   # Boosting Iterations : Number of trees or rounds
                      max_depth = 15,  # c(5, 10, 15), # Max Tree Depth : Controls the depth of the individual trees
                      eta = 0.1,       # Learning rate/ Shrinkage
                      gamma = 1,       # c(1, 2, 3), # Minimum Loss Reduction : Pseudo-regularization hyperparameter known as a Lagrangian multiplier and controls the complexity of a given tree.
                      colsample_bytree = 1, # c(0.4, 0.7, 1.0), # Subsample Ratio of Columns
                      min_child_weight = 0.5, # c(0.5, 1, 1.5), # Minimum Sum of Instance Weight
                      subsample = 0.7 ) # Subsample Percentage
                      
# Setting seed for reproducibility
set.seed(212, sample.kind="Rounding") 

# Train the model to find the best tune for XGBOOST
fit.xgboost = train(
  make.names(isFraud)~.,
  data = imbal_train_set,  
  trControl = fitControl,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  tree_method = "hist",
  metric ="ROC",
  verbose = TRUE)

# Predict with imbalanced test set
pred.xgboost <- predict(fit.xgboost,
                           imbal_test_set %>% select(-isFraud),
                           type = "prob")

# Generating ROC for the XGBOOST model
xgb_down <- roc(imbal_test_set$isFraud,
                factor(pred.xgboost$X1,ordered=TRUE),
                plot=TRUE,
                print.auc=TRUE,
                col = "#993404", 
                legacy.axes=TRUE,
                levels = c("0", "1"))

xgb_down

# Displaying Results of XGBOOST
auc_results <- rbind(auc_results,tibble(method = "XGBOOST", AUC =xgb_down$auc))
auc_results %>% knitr::kable()

# Variable importance plot for XGBOOST
vip::vip(fit.xgboost, num_features = 40, geom = "point")

# Removing unwanted objects
rm(imbal_train_set,imbal_test_set,fit.bagcart,pred.bagcart,fit.xgboost,pred.xgboost)


###########################################################
## AUC of finalized model(XGBOOST) with the validation set 
###########################################################

# Setting seed for reproducibility
set.seed(2122, sample.kind="Rounding") 

# Training the full model with the best tuning hyperparameters found.
fxgb = train(
  make.names(isFraud)~.,
  data = df,  
  trControl = fitControl,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  tree_method = "hist",
  metric ="ROC",
  verbose = TRUE)

# Predictions on the validation set
pred.xgboostval <- predict(fxgb,
                           feature_engg_fraud_det(validation_x),
                           type = "prob")

# Generating ROC for the final validation set
rocxgb_down_validation <- roc(validation_y$isFraud,
                              factor(pred.xgboostval$X1,ordered=TRUE),
                              plot=TRUE,
                              print.auc=TRUE,
                              col = "#006d2c", 
                              legacy.axes=TRUE,
                              levels = c("0", "1"))

rocxgb_down_validation

# Displaying Results of XGBOOST (Final Validation)
auc_results <- rbind(auc_results,tibble(method = "XGBOOST (Validation Set)",
                                        AUC =rocxgb_down_validation$auc))
auc_results %>% knitr::kable()

# Removing unwanted objects
rm(df,validation_x,validation_y,auc_results,rocxgb_down_validation,xgb_down,roctreebag_down)
