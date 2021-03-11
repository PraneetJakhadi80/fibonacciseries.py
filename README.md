l=6
x=0
y=1
iteration=0
if l<=0:
    print('only no greater than 0')
elif l==1:
    print('length',l)
else:
    print('length',l)
    while iteration<l:
        print(x , end=' ,')
        z=x+y
        x=y
        y=z
        iteration+=1       
