#' Plot Proportions
#' 
#' This function is useful for plotting the count and proportion of users after 
#' applying a "group by" to the data.  Error bounds are plotted, and if the 
#' group variable is randomly distributed across train/test then the 99\% of the
#' points will fall inside the confidence interval.
#' 
#' @param d The data.table, with columns cnt_col and prop_col.
#' @param cnt_col The column name of the count column in d.
#' @param prop_col The column name of the proportion column in d.
#' @param true_prop The expected split rate.
#' @param level The level of the confidence interval.
#' @param log If true, the plot (and CI) will use a log scale.
#' 
#' @return A plot
#' 

plot_proportions = function(d, cnt_col="count", prop_col="proportion", true_prop=0.5,
                            level=0.99, log=FALSE){
    stopifnot(c(cnt_col, prop_col) %in% colnames(d))
    if(!log)
        bounds = data.table(cnt=seq(min(d[[cnt_col]]), max(d[[cnt_col]]), length.out=3000))
    else
        bounds = data.table(cnt=exp(seq(log(min(d[[cnt_col]])), log(max(d[[cnt_col]])),
                                        length.out=3000)))
    bounds[, upper := true_prop + qnorm((level+1)/2) * sqrt(true_prop * (1-true_prop)/cnt)]
    bounds[, lower := true_prop + qnorm(1-(level+1)/2) * sqrt(true_prop * (1-true_prop)/cnt)]
    d[, closest_cnt := sapply(get(cnt_col), function(x){
        max(bounds$cnt[bounds$cnt <= x])
        })]
    d = merge(d, bounds, by.x="closest_cnt", by.y="cnt")
    setnames(bounds, "cnt", cnt_col)
    cat(mean(d[, proportion <= upper & proportion >= lower])*100, "% of observations fall into the ",
        level*100, "% confidence bounds\n", sep="")
    p = ggplot(d, aes_string(x=cnt_col)) +
        geom_point(aes_string(y=prop_col)) +
        geom_ribbon(data=bounds, aes(ymin=lower, ymax=upper), alpha=0.5) +
        labs(x="Counts in each bucket") +
        scale_y_continuous("Proportion train in each bucket", label=percent)
    if(log)
        p = p + scale_x_log10()
    return(p)
}
