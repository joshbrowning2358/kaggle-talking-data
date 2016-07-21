map_to_classes = function(p_male, age_mu, age_sd){
    # The 12 classes to predict are:
    # 'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+',
    # 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+'
    n = length(p_male)
    stopifnot(length(age_mu) == n)
    stopifnot(length(age_sd) == n)
    
    required_ages = c(22, 23, 26, 28, 31, 32, 38, 42)
    female_ages = c(23, 26, 28, 32, 42, 999)
    age_cumulative_probs = sapply(female_ages, function(age){
        # age + 1 since you're "28 years old" until the day you turn 29.
        mapply(pnorm, q=age + 1, mean=age_mu, sd=age_sd)
    })
    female_age_group_probs = age_cumulative_probs -
        cbind(0, age_cumulative_probs[, -ncol(age_cumulative_probs)])
    
    male_ages = c(22, 26, 28, 31, 38, 999)
    age_cumulative_probs = sapply(male_ages, function(age){
        # age + 1 since you're "28 years old" until the day you turn 29.
        mapply(pnorm, q=age + 1, mean=age_mu, sd=age_sd)
    })
    male_age_group_probs = age_cumulative_probs -
        cbind(0, age_cumulative_probs[, -ncol(age_cumulative_probs)])
    
    output = cbind(matrix(rep(1-p_male, times=6), ncol=6) * female_age_group_probs,
                   matrix(rep(p_male, times=6), ncol=6) * male_age_group_probs)
    colnames(output) = c('F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+',
                         'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')
    return(output)
}