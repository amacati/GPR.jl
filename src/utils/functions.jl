# Faster version of tr(x1*x2) by only calculating diagonals 
function product_trace(x1::AbstractMatrix, x2::AbstractMatrix, temp::AbstractMatrix)
    temp .= x1 .* x2
    return sum(temp)
end
