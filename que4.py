matrix=[[1,2,3],[5,7,5]]
transpose=[]
for i in range(len(matrix[0])):
    row=[]
    for j in range(len(matrix)):
        row.append(matrix[j][i])
    transpose.append(row)

print("Transpose:",transpose)