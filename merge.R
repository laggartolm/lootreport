library(ggplot2)

setwd('C:\\Users\\Roberto\\Dropbox\\Scripts\\Lords\\giftreport')

files <- dir()
files <- files[startsWith(files, 'WEEK')]

files <- sort(files)

#files <- files[1:2]

summ = data.frame()

for (item in files) {
  print(item)
  current <- read.csv(file = item, sep = '\t')
  week = substr(item, 12, 16)
  current["Week"] = item # not `week` anymore
  if (item == files[1]) {
    all_data <- current
  } else {
    all_data <- rbind(all_data, current)
  }
    rownames(current) <- current$Account
  current <- current["Points"]
  colnames(current) <- c(item)
  summ <- merge(summ, current, by = 0, all = TRUE)
  rownames(summ) <- summ$Row.names
  summ <- summ[2:length(summ)]

}

write.table(summ, "summary.tsv", sep="\t", col.names = NA, row.names = TRUE)
