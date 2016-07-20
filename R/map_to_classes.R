map_to_classes = function(p_male, p_female, age_mu, age_sd){
    # The 12 classes to predict are:
    # 'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+',
    # 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+'
    n = length(p_male)
    stopifnot(length(age_mu) == n)
    stopifnot(length(age_sd) == n)
    
    under23 = mapply(pnorm, q=23, mean=c(26, 29.75), sd=c(6.73, 0.7))
}