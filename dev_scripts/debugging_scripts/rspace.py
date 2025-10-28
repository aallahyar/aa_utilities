from aa_utilities.wrappers import RSpace

R = RSpace()

R("""
data <- read.table(text="
index expression mouse treat1 treat2
1 1.01 MOUSE1 NO NO
2 1.04 MOUSE2 NO NO
3 1.04 MOUSE3 NO NO
4 1.99 MOUSE4 YES NO
5 2.36 MOUSE5 YES NO
6 2.00 MOUSE6 YES NO
7 2.89 MOUSE7 NO YES
8 3.12 MOUSE8 NO YES
9 2.98 MOUSE9 NO YES
10 5.00 MOUSE10 YES YES
11 4.92 MOUSE11 YES YES
12 4.78 MOUSE12 YES YES", 
sep=" ", header=T)
print(data)

design <- model.matrix(~ treat1 + treat2, data=data)
fit <- lm(formula='expression ~ treat1 + treat2', data=data)
model_matrix <- model.matrix(fit)
model_coef <- coef(fit)
print(model_coef)
""")
print(R['model_matrix'])
print(type(R['model_matrix']))
print(R['model_coef'])
print(type(R['model_coef']))
print(R)