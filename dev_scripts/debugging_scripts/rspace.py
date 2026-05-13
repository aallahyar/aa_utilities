from aa_utilities.wrappers import RSpace

R = RSpace()

# Example: capturing R warnings
R("""
    warning("example warning message.")
""")

R("""
data <- read.table(text="
index expression mouse treat1 treat2
S1 1.01 MOUSE1 NO NO
S2 1.04 MOUSE2 NO NO
S3 1.04 MOUSE3 NO NO
S4 1.99 MOUSE4 YES NO
S5 2.36 MOUSE5 YES NO
S6 2.00 MOUSE6 YES NO
S7 2.89 MOUSE7 NO YES
S8 3.12 MOUSE8 NO YES
S9 2.98 MOUSE9 NO YES
S10 5.00 MOUSE10 YES YES
S11 4.92 MOUSE11 YES YES
S12 4.78 MOUSE12 YES YES", 
sep=" ", header=T)
rownames(data) <- data$index
print(data)

design <- model.matrix(~ treat1 + treat2, data=data)
fit <- lm(formula='expression ~ treat1 + treat2', data=data)
model_matrix <- model.matrix(fit)
model_coef <- coef(fit)
print(model_coef)
""")
print(R('fit$coef'))
# prints:
# (Intercept)    0.825833
# treat1YES      1.495000
# treat2YES      2.375000
# dtype: float64

print(R['model_matrix'])
print(type(R['model_matrix']))
print(R['model_coef'])
print(type(R['model_coef']))
print(R)

# no need to convert, R nested objects are not converted to Python objects
assert R['model_coef'].equals(R('fit$coef'))
assert R['model_matrix'].equals(R('model.matrix(fit)'))

# adding a Dict to R
R['my_dict'] = {'a': [1, 2, 3], 'b': [20, 30, 40, 50], 'c': list('ABCDEF')}
print(R('print(my_dict)'))
# prints:
# $a
# [1] 1 2 3
# $b
# [1] 20 30 40 50
# $c
# [1] "A" "B" "C" "D" "E" "F"

