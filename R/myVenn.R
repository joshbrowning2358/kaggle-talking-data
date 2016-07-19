myVenn = function(A, B){
    sizeAnotB = length(unique(A[!A %in% B]))
    sizeBnotA = length(unique(B[!B %in% A]))
    AandB = A[A %in% B]
    if(length(AandB) > 0)
        sizeAandB = length(unique(A[A %in% B]))
    else
        sizeAandB = 0
    v = venneuler(c(A=sizeAnotB, B=sizeBnotA, "A&B"=sizeAandB))
    plot(v)
}